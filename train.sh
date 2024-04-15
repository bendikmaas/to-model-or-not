#!/bin/bash
set -ex

# Only use a quarter of available resources
NUM_CPUS=$(($(nproc) / 4))
NUM_GPUS=$(($(nvidia-smi -L | wc -l) / 4)) 
TASK_NAME="${1:-Asterix}"

# Select the $NUM_GPUs with most available memory
GPU_MEM_INFO=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -n $NUM_GPUS)

# Get the available memory of the GPU with the least capacity
LAST_GPU_INFO=$(echo "$GPU_MEM_INFO" | sort -t, -k2 -n | head -n 1)
GPU_MEMORY=$(echo "$LAST_GPU_INFO" | awk -F "," '{print int($2/1024)}')  # Converts memory to GiB as an integer

# Export GPU indices for PyTorch
export CUDA_VISIBLE_DEVICES=$(echo "$GPU_MEM_INFO" | awk -F "," '{print $1}' | paste -sd "," -)

# To set these values, start the program following these instructions:
#
#  1. Leave CPU actors at 1, and increase GPU actors until GPU full according
#     to `ray satus`
#  2. Increase CPU actors until CPU full.
# ml7: 6, 8
# ml9: 2, 42
GPU_ACTORS=2
CPU_ACTORS=42

# Run script
python main.py --env "MiniGrid-LavaGapS7-v0" --case minigrid --opr train \
  --num_gpus "$NUM_GPUS" --num_cpus "$NUM_CPUS" --gpu_mem "$GPU_MEMORY" \
  --cpu_actor $CPU_ACTORS --gpu_actor $GPU_ACTORS \
  --seed 0 \
  --use_priority \
  --use_max_priority \
  --amp_type 'torch_amp' \
  --info 'MuZero-more-stable-lr-and-buffer--10k' \
  --auto_resume \
  --object_store_mem=1000000000