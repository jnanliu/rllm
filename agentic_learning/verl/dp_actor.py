import logging
import os
import numpy as np
import math

import torch
import torch.nn as nn
from transformers.tokenization_utils import PreTrainedTokenizer

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_torch_device, is_cuda_available
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input

from verl.workers.actor.dp_actor import DataParallelPPOActor as DataParallelPPOActor_


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    result_mask,
    loss_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        result_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
        clip_ratio_c: (float) default: 3.0
            The lower bound of the ratio for dual-clip PPO, See https://arxiv.org/pdf/1912.09729
        loss_agg_mode: (str) see `agg_loss`

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
        pg_clipfrac_lower: (float)
            the fraction of policy gradient loss being clipped when the advantage is negative
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.where(
        result_mask.bool(),
        torch.exp(negative_approx_kl),
        torch.exp(log_prob)
    )

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(
        ratio, 
        (1 - cliprange_low) * ratio / (ratio.detach() + 1e-6),
        (1 + cliprange_high) * ratio / (ratio.detach() + 1e-6))  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    pg_losses3 = -advantages * clip_ratio_c
    pg_losses4 = -advantages * torch.clamp_min(ratio, 0.1 * ratio / (ratio.detach() + 1e-6))
    
    
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    # Dual-clip PPO
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    clip_pg_losses3 = torch.maximum(pg_losses1, pg_losses4)

    # Remove the dual-clip PPO for now... (there's no evidence it improves performance)
    # torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_losses = torch.where(
        result_mask.bool(),
        clip_pg_losses1,
        clip_pg_losses3
    )

    # Statistics tracked for PPO.
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), loss_mask)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, loss_mask)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), loss_mask)
    
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=loss_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


class DataParallelPPOActor(DataParallelPPOActor_):
    def __init__(
        self, 
        config, 
        actor_module: nn.Module, 
        tokenizer: PreTrainedTokenizer,
        actor_optimizer: torch.optim.Optimizer = None
    ):
        super().__init__(config, actor_module, actor_optimizer)
        self.tokenizer = tokenizer

    def _forward_micro_batch(
        self, 
        micro_batch, 
        temperature, 
        calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1) if "responses" in micro_batch else -1
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        input_data = data

        self.actor_module.train()

        temperature = input_data.meta_info["temperature"]
        multi_turn = input_data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if "traj_mask" in input_data.batch:
            select_keys.append("traj_mask")
        if "result_mask" in input_data.batch:
            select_keys.append("result_mask")

            if 'is_pad_step' in input_data.non_tensor_batch:
                is_pad_step = input_data.non_tensor_batch["is_pad_step"]
                pad_step_indices = np.where(is_pad_step == True)[0]
                if len(pad_step_indices) > 0:
                    input_data.batch["advantages"][pad_step_indices] = 0

        non_tensor_select_keys = ["is_policy"]

        batch = input_data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in input_data.non_tensor_batch.keys()

        if self.config.use_dynamic_mini_batch:
            num_mini_batches = self.config.ppo_num_mini_batches
            self.config.ppo_mini_batch_size = math.ceil(input_data.batch.batch_size[0] / self.config.ppo_num_mini_batches)
            print(f"Dynamic mini batch is enabled, update ppo_mini_batch_size to {self.config.ppo_mini_batch_size}")
        else:
            num_mini_batches = input_data.batch.batch_size[0] // self.config.ppo_mini_batch_size

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
            # dataloader = input_data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        # else:
        #     dataloader = batch.split(self.config.ppo_mini_batch_size)
        dataloader = input_data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    _, micro_bsz_idx = rearrange_micro_batches(batch=mini_batch.batch, max_token_len=max_token_len)
                    micro_batches = []
                    for partition in micro_bsz_idx:
                        curr_micro_batch = []
                        for idx in partition:
                            curr_micro_batch.append(mini_batch[idx: idx + 1])
                        curr_micro_batch = mini_batch.concat(curr_micro_batch)

                        micro_batches.append(curr_micro_batch)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    # micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(self.gradient_accumulation)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        micro_batch=data, 
                        temperature=temperature, 
                        calculate_entropy=calculate_entropy
                    )

                    # filter poicy / sft
                    ind = data["is_policy"]

                    policy_loss = 0.0
                    sft_loss = 0.0
                    
                    if (ind == 1).sum() > 0:
                        responses = data["responses"][ind == 1]
                        response_length = responses.size(1)
                        attention_mask = data['attention_mask'][ind == 1]
                        if multi_turn:
                            response_mask = data["loss_mask"][ind == 1][:, -response_length:]
                        elif "traj_mask" in data:
                            response_mask = data['traj_mask'][ind == 1]
                        else:
                            response_mask = attention_mask[ind == 1, -response_length:]

                        result_mask = data["result_mask"][ind == 1]

                        old_log_prob = data["old_log_probs"][ind == 1]
                        advantages = data["advantages"][ind == 1]
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob[ind == 1],
                            advantages=advantages,
                            result_mask=result_mask,
                            loss_mask=(
                                response_mask ^ result_mask
                                if input_data.meta_info.get("mask_result", False) 
                                else response_mask
                            ),
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )
                        if entropy_coeff == 0:
                            loss_agg_mode_entropy = 'token-mean'
                        else:
                            loss_agg_mode_entropy = loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropy[ind == 1], loss_mask=response_mask, loss_agg_mode=loss_agg_mode_entropy)
                        with torch.no_grad():
                            entropy_token_mean_loss = agg_loss(loss_mat=entropy[ind == 1], loss_mask=response_mask, loss_agg_mode='token-mean')

                        # compute policy loss
                        if entropy_coeff != 0:
                            policy_loss = pg_loss - entropy_loss * entropy_coeff
                        else:
                            policy_loss = pg_loss

                        if self.config.use_kl_loss:
                            ref_log_prob = data["ref_log_prob"][ind == 1]
                            # compute kl loss
                            kld = kl_penalty(logprob=log_prob[ind == 1], ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            metrics["actor/kl_loss"] = kl_loss.detach().item()
                            metrics["actor/kl_coef"] = self.config.kl_loss_coef

                        if self.config.use_dynamic_bsz:
                            # relative to the dynamic bsz
                            policy_loss = policy_loss * ((ind == 1).sum() / self.config.ppo_mini_batch_size)
                        else:
                            policy_loss = policy_loss / self.gradient_accumulation
                        # loss.backward()

                        current_metrics = {
                            'actor/entropy_token_mean_loss': entropy_token_mean_loss.detach().item(),
                            'actor/entropy_loss': entropy_loss.detach().item(),
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                        append_to_dict(metrics, current_metrics)
                    # ====================================== SFT Loss ============================================
                    if (ind == 0).sum() > 0:
                        responses = data["responses"][ind == 0]
                        response_length = responses.size(1)
                        response_mask = data['attention_mask'][ind == 0, -response_length:]
                        
                        sft_loss = 0.5 * (
                            torch.exp(-entropy[ind == 0]).detach() * -log_prob[ind == 0] * response_mask.float()
                        ).sum() / (
                            response_mask.float().sum() + 1e-6
                        )
                        if self.config.use_dynamic_bsz:
                            # relative to the dynamic bsz
                            sft_loss = sft_loss * ((ind == 0).sum() / self.config.ppo_mini_batch_size)
                        else:
                            sft_loss = sft_loss / self.gradient_accumulation
                        # loss.backward()

                        current_metrics = {
                            "actor/sft_loss": sft_loss.detach().item()
                        }
                        append_to_dict(metrics, current_metrics)
                    # ============================================================================================
                    loss = policy_loss + sft_loss
                    if isinstance(loss, torch.Tensor):
                        loss.backward()

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)

        self.actor_optimizer.zero_grad()
        return metrics