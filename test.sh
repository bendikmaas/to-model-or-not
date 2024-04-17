set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=0

python main.py --env 'MiniGrid-LavaCrossingS7N1-' --case minigrid --opr test --seed 0 --num_gpus 1 --num_cpus 20 --force \
  --test_episodes 100 \
  --load_model \
  --amp_type 'torch_amp' \
  --model_path 'results/LavaGapS7/MuZero/img=True/av=False/seed=0/model/model_20000.p' \
  --info 'Test'
