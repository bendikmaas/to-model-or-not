python test.py --env "procgen-climber-v0" --case procgen --opr train \
  --num_gpus 2 --num_cpus 12 --gpu_mem 10 \
  --seed 0 \
  --use_priority \
  --use_max_priority \
  --amp_type 'torch_amp' \
  --info 'EfficientZero-V3' \
  --record_video \
  --auto_resume \
  --object_store_mem=1000000000