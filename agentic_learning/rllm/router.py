import asyncio
import logging

from verl.protocol import DataProto

from rllm.router.router import Router as Router

logger = logging.getLogger(__name__)


class AgenticLearningRouter(Router):

    async def generate_sequences(self, batch: DataProto, application_id: str, **sampling_params):
        kwargs = dict(
            n=self.config.actor_rollout_ref.rollout.n,
            max_tokens=self.config.actor_rollout_ref.rollout.response_length,  # Changed from max_completion_tokens
            temperature=self.config.actor_rollout_ref.rollout.temperature,
            top_p=self.config.actor_rollout_ref.rollout.top_p,
            logprobs=1,
        )

        do_sample = batch.meta_info.get("do_sample", True)
        is_validate = batch.meta_info.get("validate", False)
        if not do_sample or is_validate:
            kwargs["n"] = 1
            kwargs["temperature"] = 0

        if is_validate:
            kwargs.update(
                {
                    #'top_k': self.config.val_kwargs.top_k,
                    "top_p": self.config.actor_rollout_ref.rollout.val_kwargs.top_p,
                    "temperature": self.config.actor_rollout_ref.rollout.val_kwargs.temperature,
                    "n": 1,  # if validate, already repeat in ray_trainer
                }
            )

        if batch.meta_info.get("max_tokens", None) is not None:
            kwargs["max_tokens"] = batch.meta_info["max_tokens"]

        if batch.meta_info.get("agent_rollout", False):
            kwargs["n"] = 1

        kwargs.update(sampling_params)

        if "meta_info" in kwargs:
            if "stop" in kwargs["meta_info"]:
                kwargs["stop"] = kwargs["meta_info"]["stop"]

        address = await self.get_address(application_id)

        tasks = []
        # Bug: len(batch) is used later but batch might not have a __len__ method
        batch_size = len(batch.non_tensor_batch["formatted_prompts"])
        batch_response_ids: list[list[int]] = [[] for _ in range(batch_size)]

        for batch_index, formatted_prompt in enumerate(batch.non_tensor_batch["formatted_prompts"]):
            # For Completion API, we need to convert the conversation to a prompt string
            self.counter += 1
            tasks.append(
                self.submit_completions(  # Changed from submit_chat_completions
                    address=address,
                    model=self.model_name,
                    prompt=formatted_prompt,  # Changed from messages
                    **kwargs,
                )
            )

        # Potential blocking: asyncio.gather can block if any task takes too long
        logger.debug("Sending total requests: %s", self.counter)
        completions_list = await asyncio.gather(*tasks)
        await self.release_address(address, application_id)  # Release the address when done

        for batch_index, completions in enumerate(completions_list):
            comps = []
            for choice in completions.get("choices", []):
                token_ids = choice.get("logprobs", {}).get("tokens", [])
                token_ids = [int(t.split(":")[1]) for t in token_ids]
                # vllm
                if "finish_reason" in choice:
                    finish_reason = choice.get("finish_reason")
                    if finish_reason == "</query>":
                        token_ids += self.tokenizer.encode("</query>", add_special_tokens=False)
                # sglang
                elif "matched_stop" in choice:
                    matched_stop = choice.get("matched_stop")
                    if matched_stop == "</query>":
                        token_ids += self.tokenizer.encode("</query>", add_special_tokens=False)

                comps.append(token_ids)
            batch_response_ids[batch_index] = comps

        return await self.postprocess_batch(batch, batch_response_ids, kwargs["n"])
