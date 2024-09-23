#!/bin/bash
#SBATCH -N 1 --ntasks-per-node 8 -A llmservice_modelalignment_ppo --job-name llmservice_modelalignment_rloo_llama:gsm8k_minitron_training_actor -t 4:00:00 --dependency singleton --exclusive --gpus-per-node=8 --partition=batch_block1,batch_block3,batch_block4
#SBATCH hetjob
#SBATCH -N 4 --ntasks-per-node 8 -A llmservice_modelalignment_ppo --job-name llmservice_modelalignment_rloo_llama:gsm8k_minitron_training_rm -t 4:00:00 --dependency singleton --exclusive --gpus-per-node=8 --partition=batch_block1,batch_block3,batch_block4

RLHF_SHARED_DIR="/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo"
DATA_DIR="/lustre/fsw/portfolios/llmservice/users/abukharin/test"
WANDB_API_KEY="d5c9af701b905bfeadb7a5c7a4c2101afcbf3cc1"

NAME="minitron-8b-gsm8k-llama3.1-70BRM-lr1e-6-kl0.05"
COMMIT_ID=34205fd
# CONTAINER="${RLHF_SHARED_DIR}/containers/nemo-aligner:v2-022924-nemo-1.23.0.sqsh"
#CONTAINER="gitlab-master.nvidia.com/dl/joc/nemo-ci/main/train:pipe.16440368-x86"
CONTAINER_ACTOR="/lustre/fsw/portfolios/llmservice/users/geshen/small_model_alignment/dl+joc+nemo-ci+main+train+pipe.17556452-x86.sqsh"
CONTAINER_RM="gitlab-master.nvidia.com/dl/joc/nemo-ci/main/train:pipe.16440368-x86"
echo "Starting job at $(date '+%Y-%m-%d %H:%M:%S')"

RESULTS_DIR="${DATA_DIR}/exp/rlhf/${NAME}"
mkdir -p ${RESULTS_DIR}
NEMO_RLHF_DIR=${RESULTS_DIR}/NeMo-Aligner

pushd ${RESULTS_DIR}
if [ ! -d "${NEMO_RLHF_DIR}" ]; then
    #git clone git@github.com:NVIDIA/NeMo-Aligner.git
    git clone https://github.com/abukharin3/NeMo-Aligner.git
fi
pushd ${NEMO_RLHF_DIR}
git fetch origin
git checkout -B minitron ${COMMIT_ID} || exit 1
popd
popd

NUM_ROLLOUTS=128
NORMALIZE="True"
ACTOR_LR="1e-6"
ACTOR_GBS=512
CRITIC_GBS=64

NORMALIZE_REWARD=True
REWARD_MEAN=0.0
REWARD_STD=1

# PARAMETERS
#RM_NEMO_FILE="${DATA_DIR}/exp/rlhf/38-19_rm-15bct2-mx-hh-lr3e-6/checkpoints/megatron_gpt.nemo"
RM_NEMO_FILE="/lustre/fsw/portfolios/llmservice/users/zhilinw/models/llama31_70b_instruct_regression_helpsteer_v11_0_to_4_helpfulness_only_to_bt_weighted_shuffled_all_weights_1_epochs_constant_lr_1e-6_step_80"
ACTOR_NEMO_FILE="/lustre/fsw/portfolios/llmservice/users/geshen/share/8b_dpo-urban_3.002e-7-kl-1e-3-dpo-loss-rpo_fwd_kl-sft-weight-1e-5_megatron_gpt--val_loss=0.061-step=150-consumed_samples=38400-epoch=0/megatron_gpt--val_loss=0.061-step=150-consumed_samples=38400-epoch=0"
DATASET_DIR="${RLHF_SHARED_DIR}/data/extra_id_prefix_end_with_backslash_n_extra_id_1_jsonl"
TRAIN_DATA_PATH="/lustre/fsw/portfolios/llmservice/users/abukharin/data/minitron_gsm8k_train.jsonl"
VALID_DATA_PATH="/lustre/fsw/portfolios/llmservice/users/abukharin/data/minitron_gsm8k_val.jsonl"

MOUNTS="--container-mounts=/lustre/fsw/portfolios/llmservice/users/abukharin/data/:/lustre/fsw/portfolios/llmservice/users/abukharin/data/,${RLHF_SHARED_DIR}:${RLHF_SHARED_DIR},${RESULTS_DIR}:${RESULTS_DIR},${RM_NEMO_FILE}:${RM_NEMO_FILE},${ACTOR_NEMO_FILE}:${ACTOR_NEMO_FILE},${DATA_DIR}:${DATA_DIR},${DATA_DIR}/c/pytriton:/pytriton_cache,/lustre:/lustre"

# W&B Logging
WANDB_PROJECT="minitron"

# START HETEROGENEUS JOB 0 =======================================================
CRITIC_CONFIG_PATH="${NEMO_RLHF_DIR}/examples/nlp/gpt/conf"
CRITIC_CONFIG_NAME="inference_rm"
CRITIC_LOG_DIR="${RESULTS_DIR}/critic_results"
CRITIC_OUTFILE="${CRITIC_LOG_DIR}/critic_output_%j.log"
CRITIC_ERRFILE="${CRITIC_LOG_DIR}/critic_error_%j.err"
CRITIC_PORT=5567

mkdir -p $CRITIC_LOG_DIR

CRITIC_NAME="${NAME}_critic"

# NB: we set `attribute_weights` to only use the Helpfulness attribute

read -r -d '' cmd_critic_inference <<EOF
export WANDB_API_KEY=${WANDB_API_KEY}  \
&& cd ${NEMO_RLHF_DIR} \
&& export HYDRA_FULL_ERROR=1 \
&& export HF_TOKEN="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" \
&& export PYTRITON_HOME=/pytriton_cache \
&& export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
&& python -u examples/nlp/gpt/serve_reward_model.py \
    --config-path=${CRITIC_CONFIG_PATH} \
    --config-name=${CRITIC_CONFIG_NAME} \
    trainer.num_nodes=1 \
    trainer.devices=8 \
    ++model.tensor_model_parallel_size=4 \
    ++model.regression.num_attributes=9 \
    ++model.regression.merge_attributes=True \
    ++model.regression.attribute_weights="[0, 0, 0, 0, 1, 0, 0, 0, 0]" \
    rm_model_file=${RM_NEMO_FILE} \
    inference.port=${CRITIC_PORT}
EOF

srun --het-group=0 -o $CRITIC_OUTFILE -e $CRITIC_ERRFILE --container-image=${CONTAINER_RM} $MOUNTS bash -c "${cmd_critic_inference}" & pids[0]=$!

# END HETEROGENEUS JOB 0

sleep 30
#########################################################################################

# START HETEROGENEUS JOB 1

CONF_DIR="${NEMO_RLHF_DIR}/examples/nlp/gpt/conf"
CONFIG_NAME="gpt_reinforce_actor"

ACTOR_LOG_DIR="${RESULTS_DIR}/actor_results"
CHECKPOINT_DIR="${ACTOR_LOG_DIR}/checkpoints"
TENSOBOARD_DIR="${ACTOR_LOG_DIR}/tensorboard"

PPO_ERRFILE="${ACTOR_LOG_DIR}/actor_error_%j.err"
PPO_OUTFILE="${ACTOR_LOG_DIR}/actor_output_%j.log"

mkdir -p $ACTOR_LOG_DIR
mkdir -p $TENSOBOARD_DIR
mkdir -p $CHECKPOINT_DIR

ACTOR_NAME="${NAME}_actor"

host_critic="$(scontrol show hostnames=$SLURM_JOB_NODELIST_HET_GROUP_0 | head -n1)"


read -r -d '' cmd_ppo <<EOF
pip install immutabledict
pip install nltk
pip install langdetect
echo "import nltk;
nltk.download('all')" > download_nltk_data.py
python download_nltk_data.py

export WANDB_API_KEY=${WANDB_API_KEY} \
&& cd ${NEMO_RLHF_DIR} \
&& export HYDRA_FULL_ERROR=1 \
&& export HF_HOME="/lustre/fsw/portfolios/llmservice/users/abukharin/test/hf_home" \
&& export CUDA_LAUNCH_BLOCKING=1 \
&& export PYTRITON_HOME=/pytriton_cache \
&& export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
&& python -u examples/nlp/gpt/train_gpt_reinforce_ifeval.py \
    --config-path=${CONF_DIR} \
    --config-name=${CONFIG_NAME} \
    "model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}" \
    pretrained_checkpoint.restore_from_path=\"${ACTOR_NEMO_FILE}\" \
    exp_manager.checkpoint_callback_params.save_top_k=80 \
    exp_manager.explicit_log_dir=\"${ACTOR_LOG_DIR}\" \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name=\"${ACTOR_NAME}\" \
    exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT} \
    ++exp_manager.max_time_per_run=\"00:03:30:00\" \
    trainer.reinforce.initial_policy_kl_penalty=0.05 \
    trainer.reinforce.max_epochs=10 \
    trainer.reinforce.max_steps=313 \
    trainer.reinforce.val_check_interval=5 \
    trainer.reinforce.save_interval=10 \
    trainer.num_nodes=4 \
    trainer.devices=8 \
    ++model.tensor_model_parallel_size=4 \
    model.global_batch_size=${ACTOR_GBS} \
    model.micro_batch_size=1 \
    model.optim.lr=\"\\\$\{multiply:${ACTOR_LR},1.001\}\" \
    model.optim.sched.warmup_steps=0 \
    model.optim.sched.constant_steps=312 \
    model.optim.sched.min_lr=${ACTOR_LR} \
    model.optim.weight_decay=0.01 \
    model.reinforce.num_rollout_samples=${NUM_ROLLOUTS} \
    model.reinforce.rollout_micro_batch_size=16 \
    model.reinforce.forward_micro_batch_size=16 \
    model.reinforce.val_rollout_micro_batch_size=16 \
    model.reinforce.num_val_samples=256 \
    model.reinforce.sampling_params.end_strings="[\"<|endoftext|>\", \"<extra_id_1>\"]" \
    model.data.data_impl=jsonl \
    remote_critic_rm.reward_model.ip=${host_critic} \
    remote_critic_rm.reward_model.port=${CRITIC_PORT} \
    model.reinforce.num_rollout_per_prompt=4 \
    trainer.reinforce.baseline=\"RLOO\" \
    trainer.reinforce.ifeval_multiplier=1 \
    trainer.reinforce.rm_multiplier=1
EOF

srun --het-group=1 -o $PPO_OUTFILE -e $PPO_ERRFILE --container-image=${CONTAINER_ACTOR} $MOUNTS bash -c "${cmd_ppo}" & pids[1]=$!

# END HETEROGENEUS JOB 1


# The code below monitors the four SLURM jobs to ensure any failure forces them all to stop
# (otherwise some jobs may remain pending until they reach the cluster time limit).
all_done=false
while ! $all_done; do
    all_done=true
    for pid in "${pids[@]}"; do
        if ps -p "$pid" > /dev/null; then
            # Process is still running.
            all_done=false
        else
            # Process is no longer running => check its exit status.
            wait "$pid"
            exit_code=$?
            echo "Process $pid exited with code $exit_code at $(date '+%Y-%m-%d %H:%M:%S')"
            # Wait a bit (to get a clean stack trace in case there is one being generated), then kill the
            # remaining processes if needed.
            sleep 60
            for other_pid in "${pids[@]}"; do
                if ps -p "$other_pid" > /dev/null; then
                    echo "Killing processs $other_pid"
                    kill -9 "$other_pid"
                fi
            done
            exit $exit_code
        fi
    done

    # Sleep for a while before checking again.
    sleep 60
done

echo "Job terminated successfully"