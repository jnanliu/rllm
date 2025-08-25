set -exo pipefail

export PYTHONPATH=$PYTHONPATH:$(pwd)
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export WANDB_BASE_URL="https://api.bandw.top"
export WANDB_KEY=$6
export SWANLAB_API_KEY=$7

wandb login ${WANDB_KEY}

model=$1

timestamp=$(date +"%Y%m%d_%H%M%S")

project_name=AgenticLearning
exp_name=${project_name}-NativeMath-${model#*/}-${timestamp}

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

if [ $MLP_HOST == $MLP_WORKER_0_HOST ]
then
ray start --head --port=8266 &
sleep 10

python3 -m agentic_learning.scripts.math.train \
    data.train_batch_size=${train_batch_size} \
    data.val_batch_size=256 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.model.path=${model} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
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
    actor_rollout_ref.actor.grad_norm_threshold=10 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload}  \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.chat_scheduler=verl.schedulers.completions_scheduler.CompletionsScheduler \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.8 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${gpu_memory_utilization}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${gen_tp}" \
    actor_rollout_ref.rollout.max_model_len="${max_model_len}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${ref_ppo_max_token_len}" \
    actor_rollout_ref.rollout.n="${num_generation_per_prompt}" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_num_batched_tokens="${actor_ppo_max_token_len}" \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
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
    trainer.total_training_steps=250 \
    trainer.resume_mode=disable \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.logger=["console","swanlab"] \
    trainer.nnodes=${nnodes} \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.log_val_generations=100 \
    trainer.default_local_dir=checkpoints/${project_name}/${exp_name} \
    trainer.rollout_data_dir=checkpoints/${project_name}/${exp_name}/rollout_data \
    trainer.validation_data_dir=checkpoints/${project_name}/${exp_name}/validation_data \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    agent.max_steps=20 \
    agent.use_stepwise_advantage=False \
    agent.overlong_filter=True \
    +agent.max_turns=5 \
    +agent.mask_result=False \
    +agent.enable_interaction=False \
    +env.model_name="Qwen/Qwen3-30B-A3B-Instruct-2507" \
    +env.base_url=["https://sd2egkbccck1fc4k45t20.apigateway-cn-beijing.volceapi.com/v1"] \

else
# init worker
sleep 10
ray start --address=$MLP_WORKER_0_HOST:8266 &
bash -lc -- "sleep infinity"
fi