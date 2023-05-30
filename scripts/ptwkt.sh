#!/usr/bin/env bash
# Pretrain on wikipedia paragraphs
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
# pip install -r requirements.txt
# conda install -y pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
# conda install -y pyg -c pyg 

# test cuda
python -c "import torch; print('device_count:', torch.cuda.device_count())"
python -c "import torch_geometric; print('torch_geometric version:', torch_geometric.__version__)"

export TOKENIZERS_PARALLELISM=true

# run pretrain
python  train.py --pretrain --config configs/pretrain.yaml --config-profile pretrain_wikitop --run-name ptwkt-0pf
