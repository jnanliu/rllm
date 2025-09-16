import asyncio
import time
import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import openai
from openai.types import Completion
from transformers import PreTrainedTokenizerBase
from tqdm import tqdm
import uuid

from rllm.environments import BaseEnv
from rllm.agents.agent import Action, Trajectory
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.agents.utils import get_recent_assistant_user_messages
from rllm.misc import colorful_print
from rllm.environments.env_utils import compute_mc_return, compute_trajectory_reward
from rllm.parser.chat_template.parser import ChatTemplateParser

from agentic_learning.rllm.agent import AgenticLearningAgent
from agentic_learning.rllm.router import AgenticLearningRouter

logger = logging.getLogger(__name__)


def convert_messages_to_tokens_and_masks(
    messages: list[dict[str, str]], 
    tokenizer: PreTrainedTokenizerBase, 
    parser: ChatTemplateParser, 
    contains_first_msg=False, 
    contains_generation_msg=False
):
    """
    Converts multiple messages to tokens and masks.
    contains_first_msg flag and contains_generaiton_msg flag are used to indicate whether the conversation is for beginning or contains the generation.
    The first and last message is assumed to be the special message respectively

    Args:
        messages (List[Dict]): The messages to convert.
        tokenizer: The tokenizer to use.
        contains_first_msg (bool): Whether the first message is a special message.
        contains_generation_msg (bool): Whether the last message is a special message.

    Returns:
        Tuple[List[int], List[int]]: A tuple containing all tokens and all masks.
    """
    all_msg_tokens = []
    all_msg_masks = []

    def _convert_message_to_tokens_and_masks(msg, first_msg=False, generation_msg=False):
        msg_text = parser.parse([msg], add_generation_prompt=generation_msg, is_first_msg=first_msg)

        # Remove the assistant token since it is contained in previous message as generation prompt
        if msg["role"] == "assistant":
            assert msg_text.startswith(parser.assistant_token), f"Expected assistant token {parser.assistant_token} but got {msg_text}"
            msg_text = msg_text.replace(parser.assistant_token, "")

        msg_tokens = tokenizer.encode(msg_text, add_special_tokens=False)
        mask_value = 1 if msg["role"] == "assistant" else 0
        msg_mask = [mask_value] * len(msg_tokens)

        return msg_tokens, msg_mask

    for i, msg in enumerate(messages):
        msg_tokens, msg_mask = _convert_message_to_tokens_and_masks(msg, first_msg=(contains_first_msg and i == 0), generation_msg=(contains_generation_msg and i == len(messages) - 1))
        all_msg_tokens.extend(msg_tokens)
        all_msg_masks.extend(msg_mask)

    return all_msg_tokens, all_msg_masks


class AgenticLearningAgentExecutionEngine(AgentExecutionEngine):
    def __init__(
        self,
        engine_name="openai",
        tokenizer=None,
        rollout_engine=None,
        chat_parser=None,
        n_parallel_agents=1,
        trajectory_timeout=None,
        gamma=0.2,
        api_retries=3,
        retry_limit=3,
        max_steps=5,
        max_response_length=8192,
        max_prompt_length=1024,
        config=None,
        agent_class=None,
        env_class=None,
        agent_args=None,
        rollout_engine_args=None,
        env_args=None,
        max_workers=64,
        enforce_max_prompt_length=False,  # If enabled, applies max_prompt check per step
        overlong_filter=False,  # Filter for overlong trajectories (i.e. TRUNCATION, MAX_STEPS, TIMEOUT)
        **kwargs,
    ):
        if agent_args is None:
            agent_args = {}
        if rollout_engine_args is None:
            rollout_engine_args = {}
        if env_args is None:
            env_args = {}

        self.config = config
        self.rollout_engine = rollout_engine
        self.tokenizer = tokenizer
        self.engine_name = engine_name
        self.n_parallel_agents = n_parallel_agents
        self.overlong_filter = overlong_filter

        # For interaction
        self.gamma = gamma
        self.retry_limit = retry_limit
        self.api_retries = api_retries
        self.max_steps = max_steps
        self.max_response_length = max_response_length
        self.max_prompt_length = max_prompt_length
        self.enforce_max_prompt_length = enforce_max_prompt_length

        self.agent_class = agent_class
        self.agent_args = agent_args
        self.env_class = env_class
        self.env_args = env_args

        self.agents = [None for _ in range(n_parallel_agents)]
        self.envs = [None for _ in range(n_parallel_agents)]

        self.trajectory_timeout = trajectory_timeout
        if not trajectory_timeout:
            self.trajectory_timeout = int(1e9)

        if env_class is not None:
            assert env_class.is_multithread_safe(), "Environment must be multithread safe for async engine"
        # rollout engine args
        self.rollout_engine_args = rollout_engine_args
        self.sampling_params = kwargs.get("sampling_params", {})

        assert self.engine_name in ["openai", "verl"], "Currently only openai and verl are supported as rollout engine"
        if self.engine_name == "openai":
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(**self.rollout_engine_args)
            # Disable httpx INFO logs that show HTTP requests
            logging.getLogger("httpx").setLevel(logging.WARNING)
        elif self.engine_name == "verl":
            # All generation is done via scheduler. Currently only works for verl
            self.server_addresses = getattr(self.rollout_engine, "server_addresses", [])
            self.router = AgenticLearningRouter(
                config=self.config, 
                tokenizer=self.tokenizer, 
                addresses=self.server_addresses
            )

        # Create a thread pool executor for environment interactions (i.e. step, reset, close)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        if chat_parser is None:
            self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer, disable_thinking=kwargs.get("disable_thinking", False))
        else:
            self.chat_parser = chat_parser

    def _convert_prompt_verl(self, prompts, **kwargs):
        """
        Given a list of prompts in Chat template, convert to DataProto format in veRL

        Args:
            prompts: List of prompts to convert
            **kwargs: Additional arguments

        Returns:
            DataProto object containing the converted prompts
        """
        from verl.protocol import DataProto, union_two_dict
        from verl.utils.model import compute_position_id_with_mask
        from verl.utils.torch_functional import pad_sequence_to_length

        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        formatted_prompts = []
        for prompt in prompts:
            prompt_text = self.chat_parser.parse(
                prompt, 
                add_generation_prompt=True if prompt[-1]["role"] != "assistant" else False, 
                is_first_msg=True
            ) 
            if prompt[-1]["role"] == "assistant":
                if hasattr(self.chat_parser, "eot_token"):
                    prompt_text = prompt_text[:-len(self.chat_parser.eot_token)]
            formatted_prompts.append(prompt_text)

        # Tokenize the final processed strings
        inputs = self.tokenizer(
            formatted_prompts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        self.tokenizer.padding_side = old_padding_side

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # pad to max sizes
        input_ids = pad_sequence_to_length(
            input_ids, 
            max_seq_len=self.max_prompt_length, 
            pad_token_id=self.tokenizer.pad_token_id, 
            left_pad=True
        )
        attention_mask = pad_sequence_to_length(
            attention_mask, 
            max_seq_len=self.max_prompt_length, 
            pad_token_id=0, 
            left_pad=True
        )
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        data = DataProto.from_dict(batch_dict)
        data.non_tensor_batch["formatted_prompts"] = np.array(formatted_prompts)

        # original_batch contains the extra info needed for generation
        if "meta_info" in kwargs and kwargs["meta_info"]:
            meta_info = kwargs["meta_info"]
            # only use the original_batch's meta_info since tensor_batch is from batch_dict and non_tensor_batch is not neeeded
            data.meta_info = union_two_dict(data.meta_info, meta_info)

        return data

    async def _get_openai_async(self, prompt, _, **kwargs):
        """
        Get action from OpenAI API asynchronously with retry logic.

        Args:
            prompt: The input prompt in text format for completions API
            application_id: Unique identifier for the application (unused for OpenAI)
            **kwargs: Additional arguments to pass to the OpenAI API

        Returns:
            The response from OpenAI API
        """

        async def get_response(prompt_text: str):
            retries = self.api_retries
            while retries > 0:
                try:
                    response = await self.client.completions.create(
                        prompt=prompt_text,
                        timeout=3600,
                        **self.sampling_params,
                        **kwargs,
                    )
                    return response
                except openai.RateLimitError:
                    retries -= 1
                    if retries == 0:
                        return "Error: Rate limit reached and retries exhausted."
                    logger.info("Sleep for 5 seconds for API limit.")
                    await asyncio.sleep(5)
                except Exception as e:
                    logger.error("Error: %s", e)
                    return f"Error processing content: {e}"

        # If prompt is in chat format, convert it to text format
        prompt_text = prompt
        if isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt):
            prompt_text = self.chat_parser.parse(
                prompt, 
                add_generation_prompt=True if prompt[-1]["role"] != "assistant" else False,
                is_first_msg=True
            )
            if prompt[-1]["role"] == "assistant":
                if hasattr(self.chat_parser, "eot_token"):
                    prompt_text = prompt_text[:-len(self.chat_parser.eot_token)]

        response = await get_response(prompt_text)
        if isinstance(response, Completion):
            response = response.choices[0].text
            if hasattr(response.choices[0], "finish_reason"):
                finish_reason = response.choices[0].finish_reason
                if finish_reason == "</query>":
                    response += "</query>"
            elif hasattr(response.choices[0], "matched_stop"):
                matched_stop = response.choices[0].matched_stop
                if matched_stop == "</query>":
                    response += "</query>"
        return response

    async def run_agent_trajectory_async(self, idx, application_id, seed=0, mode="Text", **kwargs):
        """Run a single agent's trajectory asynchronously"""
        agent = self.agents[idx]
        env = self.envs[idx]
        # env_id = env.env_id

        termination_reason = None
        prompt_token_len = 0
        prompt_tokens = []
        response_token_len = 0
        response_tokens = []
        response_masks = []
        total_time = 0.0
        reward_time = None
        llm_time = 0.0
        env_time = 0.0
        reward = 0.0

        # for step return
        episode_steps = []

        # Reset environment with the task using the executor
        loop = asyncio.get_event_loop()
        observation, info = await loop.run_in_executor(self.executor, env.reset)
        info["max_steps"] = self.max_steps

        # Reset agent
        agent.reset()
        # Update agent internal state from environment.
        agent.update_from_env(
            observation=observation,  # Raw observation from environment
            reward=0.0,
            done=False,
            info=info,
        )

        messages = agent.chat_completions
        prompt_tokens, _ = convert_messages_to_tokens_and_masks(
            messages, 
            tokenizer=self.tokenizer, 
            parser=self.chat_parser, 
            contains_first_msg=True, 
            contains_generation_msg=True
        )
        prompt_token_len = len(prompt_tokens)
        # Note, this should never happen!
        if prompt_token_len > self.max_prompt_length:
            agent.reset()
            raise Exception(f"Trajectory {idx}: initial prompt length {prompt_token_len} already exceeded max_prompt_length {self.max_prompt_length}, retrying")

        for step_idx in range(self.max_steps):
            # Get action from agent
            prompt_messages = agent.chat_completions.copy()
            # Max remaining tokens left for the response
            # For enforced max prompt at each step, no need to deduct here
            if not self.enforce_max_prompt_length:
                max_tokens = self.max_response_length - response_token_len
            else:
                max_tokens = self.max_response_length

                # since max prompt is enforced, we filter out too long prompts.
                prompt_str = self.chat_parser.parse(
                    prompt_messages, add_generation_prompt=True, is_first_msg=True)
                prompt_len = len(self.tokenizer.encode(prompt_str, add_special_tokens=False))
                if prompt_len > self.max_prompt_length:
                    termination_reason = "PROMPT_TRUNCATION"
                    break

            kwargs["max_tokens"] = max_tokens

            start_time = time.time()
            response = await self.get_model_response(prompt_messages, application_id, **kwargs)
            delta_time = time.time() - start_time
            llm_time += delta_time
            total_time += delta_time

            # Update steps
            prompt_response_pair = {
                "prompt": self.chat_parser.parse(
                    prompt_messages, 
                    add_generation_prompt=True, 
                    is_first_msg=True
                ),
                "response": response,
            }
            episode_steps.append(prompt_response_pair)

            # Update agent with model response
            action: Action = agent.update_from_model(response, tokenizer=self.tokenizer)
            action = action.action       

            # Take step in environment using the executor
            start_time = time.time()

            try:
                next_observation, reward, done, info = await asyncio.wait_for(
                    loop.run_in_executor(self.executor, env.step, action), 
                    timeout=(self.trajectory_timeout - total_time)
                )
            except asyncio.TimeoutError:
                termination_reason = "ENV_TIMEOUT"
                if step_idx == 0:
                    colorful_print(f"Warning: Trajectory {idx} completed due to: {termination_reason} before able to perform 1 complete action. This might cause unexpected behavior. Consider increasing trajectory timeout limit.\n", "red")
                reward = 0

                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            delta_time = time.time() - start_time
            env_time += delta_time
            total_time += delta_time
            info["max_steps"] = self.max_steps
            info["cur_tokens"] = response_token_len

            # Update agent internal state.
            agent.update_from_env(
                observation=next_observation,
                reward=reward,
                done=done,
                info=info,
                tokenizer=self.tokenizer
            )

            cur_step = agent.get_current_state()
            cur_step.reward = reward
            cur_step.done = done
            cur_step.info.update(info)

            # Update repsonse token length
            response_token_len = len(agent.response_tokens)
            # Reached maximum number of tokens for the trajectory
            if not self.enforce_max_prompt_length and response_token_len >= self.max_response_length:
                response_tokens = agent.response_tokens[:self.max_response_length]
                response_masks = [1] * len(response_tokens)
                response_masks = agent.response_mask[:self.max_response_length]

                cur_step = agent.get_current_state()
                if response_token_len > self.max_response_length:
                    cur_step.reward = 0.0
                cur_step.done = True
                termination_reason = "TRUNCATION"

                break
            else:
                response_tokens = agent.response_tokens
                response_masks = agent.response_mask

            if total_time >= self.trajectory_timeout:
                termination_reason = "TIMEOUT"
                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            # Check if episode is done
            if done:
                termination_reason = "ENV_DONE"
                break

            if step_idx == self.max_steps - 1:
                termination_reason = "MAX_STEPS"

        masked_out = False
        if self.overlong_filter:
            if (
                termination_reason == "TRUNCATION" or 
                termination_reason == "MAX_STEPS" or 
                termination_reason == "TIMEOUT"
            ):
                # Mask out the entire response for overlong trajectories if the reward is 0.
                response_masks = [0] * len(response_masks)
                masked_out = True

        if hasattr(env, "compute_final_reward") and not masked_out:
            cur_step = agent.get_current_state()
            start_time = time.time()
            reward = await loop.run_in_executor(self.executor, env.compute_final_reward)
            reward_time = time.time() - start_time
            cur_step.reward = reward
    
        # Closing environment using the executor.
        await loop.run_in_executor(self.executor, env.close)
        # if termination_reason:
        #     if reward > 0:
        #         color = "green"
        #     else:
        #         color = "yellow"
        #     colorful_print(
        #         f"Trajectory {idx} completed due to: {termination_reason}. Reward is {reward}. \n",
        #         color,
        #     )
        #     if masked_out:
        #         colorful_print(f"Trajectory {idx} is masked out due to overlong filter.", "red")

        trajectory: Trajectory = agent.trajectory
        # Aggregate final trajectory statistics
        compute_trajectory_reward(trajectory)
        compute_mc_return(trajectory, gamma=self.gamma)

        if mode == "Text":
            return trajectory
        # ================================== Add Knowledge ===============================
        elif mode == "Token":
            agent: AgenticLearningAgent
            
            assert len(response_masks) == len(response_tokens), print(len(response_masks), len(response_tokens))

            token_result = {
                "prompt_tokens": torch.tensor(prompt_tokens, dtype=torch.long),
                "response_tokens": torch.tensor(response_tokens, dtype=torch.long),
                "response_masks": torch.tensor(response_masks, dtype=torch.long),
                # "knowledge_tokens": all_knowledge_tokens_list,
                # "knowledge_masks": all_knowledge_masks_list,
                "knowledge_messages": agent.knowledge_messages,
                "trajectory_reward": trajectory.reward,
                "idx": env.idx,
                "termination_reason": termination_reason,
                "chat_completions": agent.chat_completions,
                "metrics": {
                    # Total number of steps taken in the trajectory
                    "steps": len(trajectory.steps),
                    # Time to calculate reward
                    "reward_time": reward_time,
                    # Total time spent in environment execution (env.step)
                    "env_time": env_time,
                    # Time to calculate response tokens
                    "llm_time": llm_time,
                    # Total time spent in the trajectory
                    "total_time": total_time,
                },
            }
            return token_result
        # ==================================================================================
        elif mode == "Conversation":
            return agent.chat_completions
        elif mode == "Step":
            steps_result = {
                "steps": episode_steps,
                "trajectory_reward": trajectory.reward,
                "idx": env.idx,
                "mc_returns": [step.mc_return for step in trajectory.steps][: len(episode_steps)],
            }
            return steps_result

    async def trajectory_generator(self, reset_seed=0, timing_raw=None, mode="Text", **kwargs):
        if timing_raw is None:
            timing_raw = {}
        assert all(env is not None and isinstance(env, BaseEnv) for env in self.envs), "All environments must be inheriting from BaseEnv"
        assert all(env.is_multithread_safe() for env in self.envs), "All environments must be multithread safe for async engine"  # type: ignore
        max_concurrency = self.n_parallel_agents
        self.executor = ThreadPoolExecutor(max_workers=max_concurrency)

        if self.engine_name == "verl":
            self.rollout_engine.wake_up()

        async def launch_one_trajectory_task(env_idx: int):
            try:
                application_id = str(uuid.uuid4())
                result = await self.run_agent_trajectory_with_retry(
                    idx=env_idx,
                    application_id=application_id,
                    seed=reset_seed,
                    mode=mode,
                    **kwargs,
                )
            except Exception as e:
                import traceback

                traceback.print_exc()
                raise e
            return result

        # Create all N conceptual tasks. Their execution will be throttled by the semaphore
        # and the availability of agent/env indices.
        tasks_to_run = [launch_one_trajectory_task(i) for i in range(len(self.envs))]

        tasks_completed = 0
        progress_bar = tqdm(total=len(tasks_to_run), desc="Rollout Progress") 
        for coro in asyncio.as_completed(tasks_to_run):
            try:
                result = await coro
                tasks_completed += 1
                if "termination_reason" in result:
                    termination_reason = result.pop("termination_reason")
                    progress_bar.set_postfix_str(f'Trajectory {result["idx"]} completed due to: {termination_reason}')
                elif "idx" in result:
                    progress_bar.set_postfix_str(f'Trajectory {result["idx"]} completed')
                progress_bar.update(1)
                # colorful_print(f"Number of Trajectories {tasks_completed}/{len(self.envs)} completed", "cyan")
                yield result
            except Exception as e:
                raise e
        
        progress_bar.close()

        if self.engine_name == "verl":
            self.rollout_engine.sleep()

        self.executor.shutdown(wait=False, cancel_futures=True)