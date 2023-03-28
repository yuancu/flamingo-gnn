#!/usr/bin/env bash
# Pretrain on wikipedia paragraphs (selected 100k articles)
#
#SBATCH --job-name ptwkt
#SBATCH --output=R-%x.%j.out
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1

# debug info
hostname
which python
nvidia-smi

# venv
source ~/miniconda3/bin/activate dragon

# test cuda
python -c "import torch; print('device_count:', torch.cuda.device_count())"
python -c "import torch_geometric; print('torch_geometric version:', torch_geometric.__version__)"

export TOKENIZERS_PARALLELISM=true

# run pretrain
python  train.py --pretrain --config configs/lmgnn.yaml --config-profile pretrain_wikitop_xl --run-name ptwktxl
