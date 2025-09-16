set -exo pipefail

cd /mnt/shared-storage-user/liujunnan/project/AgenticLearning/
source ~/.bashrc
conda activate agentic

export PYTHONPATH=$PYTHONPATH:$(pwd)
export SWANLAB_MODE="local"

project_name=short-cot-distillation
experiment_name=qwen3-30b_2_qwen3-4b

python3 -m torch.distributed.run --standalone --nnodes=$1 --nproc_per_node=$2 \
    agentic_learning/verl/fsdp_sft_trainer.py \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=16384 \
    data.truncation=right \
    data.train_files=/mnt/shared-storage-user/liujunnan/datasets/sft_train.parquet \
    data.val_files= \
    data.prompt_key="prompt" \
    data.prompt_dict_keys=[] \
    data.response_key="response" \
    data.response_dict_keys=[] \
    model.partial_pretrain=Qwen/Qwen2.5-7B-Instruct \
    model.trust_remote_code=true \
    model.enable_gradient_checkpointing=true \
    model.use_liger=True \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=True \
    trainer.total_epochs=3 \
    trainer.default_local_dir=checkpoints/${project_name}/${experiment_name} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.logger='["console","mlflow"]' \
    optim.lr=5e-5
