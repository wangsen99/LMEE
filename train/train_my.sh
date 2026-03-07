set -x
export PYTHONUNBUFFERED=1
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/lmee_config.yaml \
    data.train_files=../data/LMEE-train/train_data.parquet \
    data.val_files=../data/LMEE-train/train_data.parquet \
    data.image_key=images \
    data.answer_key=next_action \
    data.data_path=../data/LMEE-train/task_train/ \
    data.format_prompt=./examples/format_prompt/nav.jinja \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=3B_MEM \
    trainer.n_gpus_per_node=8 \
    trainer.logger=["console"] \
    trainer.save_freq=10 \
    trainer.save_limit=20 \
    trainer.load_checkpoint_path=null \
    trainer.save_checkpoint_path=./ckpt/ \
    worker.actor.global_batch_size=64 \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.actor.optim.lr=1e-6 \
    worker.actor.optim.lr_warmup_ratio=0.6 \
    worker.rollout.gpu_memory_utilization=0.7 \
    worker.rollout.n=5 \
    data.rollout_batch_size=128 \
    data.val_batch_size=32 \
    worker.reward.reward_type=llm_batch_Memory \
    worker.reward.double_reward=True \
    worker.reward.reward_function=./examples/reward_function/nav.py:compute_score