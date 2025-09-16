from typing import Any
import multiprocessing
from functools import partial

import ray
from datasets import Dataset

from rllm.trainer.env_agent_mappings import AGENT_CLASS_MAPPING, ENV_CLASS_MAPPING
from verl.trainer.ppo.reward import get_custom_reward_fn, default_compute_score

from agentic_learning.verl.ppo_trainer import AgenticLearningPPOTrainer
from agentic_learning.verl.reward_manager import NaiveRewardManager


def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    compute_score = get_custom_reward_fn(config)
    final_compute_score = compute_score

    if compute_score is None:
        sandbox_config = config.reward_model.get("sandbox_fusion")
        sandbox_url = sandbox_config.get("url") if sandbox_config else None
        if sandbox_url:
            sandbox_manager = multiprocessing.Manager()
            _concurrent_semaphore = sandbox_manager.Semaphore(
                sandbox_config.get("max_concurrent", 64)
            )
            final_compute_score = partial(
                default_compute_score, 
                sandbox_fusion_url=sandbox_url, 
                concurrent_semaphore=_concurrent_semaphore
            )
        else:
            final_compute_score = default_compute_score

    return NaiveRewardManager(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=final_compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def train_agent(config, agent_class=None, env_class=None, agent_args=None, env_args=None):
    # print initial config
    from pprint import pprint

    from omegaconf import OmegaConf

    from verl.utils.fs import copy_local_path_from_hdfs

    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer

    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    # processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

    if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
        assert config.critic.strategy in ["fsdp", "fsdp2"]
        from verl.single_controller.ray import RayWorkerGroup
        from agentic_learning.verl.fsdp_workers import AsyncActorRolloutRefWorker, ActorRolloutRefWorker
        from verl.workers.fsdp_workers import CriticWorker

        actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup
    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(max_concurrency=2048)(actor_rollout_cls),
        Role.Critic: ray.remote(CriticWorker),
    }

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }

    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        mapping[Role.RefPolicy] = global_pool_id

    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1)
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    if env_class is None:
        env_class = ENV_CLASS_MAPPING[config.env.name]
    if agent_class is None:
        agent_class = AGENT_CLASS_MAPPING[config.agent.name]

    env_args = env_args or {}
    agent_args = agent_args or {}
    if config.env.get("env_args") is not None:
        env_args.update(config.env.get("env_args"))
    if config.agent.get("agent_args") is not None:
        agent_args.update(config.agent.get("agent_args"))

    trainer = AgenticLearningPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
        env_class=env_class,
        agent_class=agent_class,
        env_args=env_args,
        agent_args=agent_args,
    )

    trainer.init_workers()
    trainer.fit_agent()


class AgenticLearningAgentTrainer:
    """
    A wrapper class that allows users to easily train custom agents with custom environments
    without having to directly interact with the underlying training infrastructure.
    """

    def __init__(
        self,
        agent_class: type,
        env_class: type,
        agent_args: dict[str, Any] | None = None,
        env_args: dict[str, Any] | None = None,
        config: dict[str, Any] | list[str] | None = None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
    ):
        """
        Initialize the AgentTrainer.

        Args:
            agent_class: The custom agent class to use for training
            env_class: The custom environment class to use for training
            config: Configuration overrides to apply to the default config
                   Can be a dictionary with dot notation keys (e.g., {"data.train_batch_size": 8})
                   or a list of strings in the format "key=value" (e.g., ["data.train_batch_size=8"])
            train_dataset: Optional train dataset to use
            val_dataset: Optional validation dataset to use
            agent_args: Optional arguments to pass to the agent class
            env_args: Optional arguments to pass to the environment class
        """
        self.agent_class = agent_class
        self.env_class = env_class
        self.agent_args = agent_args or {}
        self.env_args = env_args or {}

        self.config = config

        if train_dataset is not None and self.config is not None and hasattr(self.config, "data"):
            self.config.data.train_files = train_dataset.get_verl_data_path()
        if val_dataset is not None and self.config is not None and hasattr(self.config, "data"):
            self.config.data.val_files = val_dataset.get_verl_data_path()

    def train(self):
        if not ray.is_initialized():
            ray.init(
                runtime_env={
                    "env_vars": {
                        "TOKENIZERS_PARALLELISM": "true", 
                        "NCCL_DEBUG": "WARN", 
                        "WANDB_BASE_URL": "https://api.bandw.top"
                    }
                }
            )

        ray.get(
            train_agent.remote(
                self.config, 
                self.agent_class, 
                self.env_class, 
                self.agent_args, 
                self.env_args
            )
        )

