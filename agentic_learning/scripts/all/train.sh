set -exo pipefail

cd /mnt/shared-storage-user/liujunnan/project/AgenticLearning/
source ~/.bashrc
conda activate agentic

export PYTHONPATH=$PYTHONPATH:$(pwd)
export SWANLAB_MODE="local"
export VLLM_USE_V1=1
export RAY_memory_monitor_refresh_ms=0

model=$1

timestamp=$(date +"%Y%m%d_%H%M%S")

project_name=RL
exp_name=${project_name}-All-${model#*/}-${timestamp}

# grpo related
adv_estimator=grpo
use_kl_loss=False
kl_coef=0.0
kl_loss_coef=0.0
clip_ratio_min=0.2
clip_ratio_max=0.28

# data related
train_batch_size=256
max_prompt_length=$((1024 * 4))
max_response_length=$((1024 * 16))
max_model_len=$((max_prompt_length + max_response_length))

nnodes=$2
n_gpus_per_node=$3

use_dynamic_bsz=True
actor_ppo_max_token_len=$((2 * max_prompt_length + 2 * max_response_length))
ref_ppo_max_token_len=$((2 * max_prompt_length + 2 * max_response_length))

offload=False

sp=1

gen_tp=$4
gpu_memory_utilization=$5
num_generation_per_prompt=8

python3 -m agentic_learning.scripts.all.train \
    data.train_batch_size=${train_batch_size} \
    data.val_batch_size=256 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.model.path=${model} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.checkpoint.contents=[] \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_min} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_max} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload}  \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.chat_scheduler=verl.schedulers.completions_scheduler.CompletionsScheduler \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${gpu_memory_utilization}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${gen_tp}" \
    actor_rollout_ref.rollout.max_model_len="${max_model_len}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${ref_ppo_max_token_len}" \
    actor_rollout_ref.rollout.n="${num_generation_per_prompt}" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_num_batched_tokens="${actor_ppo_max_token_len}" \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    reward_model.enable=False \
    reward_model.reward_manager=naive \
    custom_reward_function.path=$(realpath agentic_learning/verl/reward.py) \
    algorithm.adv_estimator="${adv_estimator}" \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.kl_ctrl.kl_coef="${kl_coef}" \
    algorithm.mask_truncated_samples=True \
    algorithm.clip_advantages=False \
    trainer.val_before_train=True \
    trainer.total_epochs=1 \
    trainer.total_training_steps=300 \
    trainer.resume_mode=disable \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.logger=["console","mlflow"] \
    trainer.nnodes=${nnodes} \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.log_val_generations=100 \
    trainer.default_local_dir=checkpoints/${project_name}/${exp_name} \
    trainer.rollout_data_dir=checkpoints/${project_name}/${exp_name}/rollout_data \
    trainer.validation_data_dir=checkpoints/${project_name}/${exp_name}/validation_data \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    agent.max_steps=20 \
    agent.use_stepwise_advantage=False \
    agent.overlong_filter=True \
    +agent.max_turns=3 \
    +agent.mask_result=False \
    +agent.enable_interaction=True \
    +env.model_name="Qwen/Qwen3-30B-A3B-Instruct-2507" \
    +env.base_url=["http://100.96.234.172:7878/v1","http://100.96.234.191:8080/v1"] \
