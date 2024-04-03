#!/bin/bash
set -ex
NUM_CPUS=$(($(nproc) / 4))
NUM_GPUS=$(($(lspci | grep -i nvidia | wc -l) / 4)) 
TASK_NAME="${1:-Asterix}"

# Select the GPUs with most available memory
GPU_MEM_INFO=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -n $NUM_GPUS)
export CUDA_VISIBLE_DEVICES=$(echo "$GPU_MEM_INFO" | awk -F "," '{print $1}' | paste -sd "," -)

# To set these values, start the program following these instructions:
#
#  1. Leave CPU actors at 1, and increase GPU actors until GPU full according
#     to `ray satus`
#  2. Increase CPU actors until CPU full.
GPU_ACTORS=6
CPU_ACTORS=40

python main.py --env "MiniGrid-LavaGapS7-v0" --case minigrid --opr train \
  --num_gpus "$NUM_GPUS" --num_cpus "$NUM_CPUS" --gpu_mem 10 \
  --cpu_actor $CPU_ACTORS --gpu_actor $GPU_ACTORS \
  --seed 0 \
  --use_priority \
  --use_max_priority \
  --amp_type 'torch_amp' \
  --info 'MuZero-baseline' \
  --auto_resume \
  --object_store_mem=1000000000
