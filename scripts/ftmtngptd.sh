#!/usr/bin/env bash
# Finetune on Multilingual Complex Wikidata Questionsn with dummy graph with pretrained GNN
#
#SBATCH --job-name ftcwqngptd
#SBATCH --output=runs/R-%x.%j.out
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --signal=SIGUSR1@90

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
export WANDB__SERVICE_WAIT=300

# run pretrain
python -u train.py --finetune --config configs/lmgnn.yaml --config-profile finetune_mintaka_no_graph_pretrained --run-name ftmtngptd