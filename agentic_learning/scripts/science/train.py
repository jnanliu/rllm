import hydra

from rllm.data.dataset import DatasetRegistry

from agentic_learning.rllm.agent import AgenticLearningAgent
from agentic_learning.rllm.environment import AgenticLearningEnv
from agentic_learning.rllm.agent_trainer import AgenticLearningAgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("openscience2-20k", "train")
    test_dataset = DatasetRegistry.load_dataset("gpqa-diamond", "test")

    agent_args = {
        "enable_interaction": config.agent.enable_interaction,
        "max_turns": config.agent.max_turns
    }
    env_args = {
        "max_turns": config.agent.max_turns,
        "model_name": config.env.model_name,
        "base_url": config.env.base_url,
        "api_key": "EMPTY",
        "max_retries": 3
    }
    trainer = AgenticLearningAgentTrainer(
        agent_class=AgenticLearningAgent,
        agent_args=agent_args,
        env_class=AgenticLearningEnv,
        env_args=env_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
