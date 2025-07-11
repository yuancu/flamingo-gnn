#!/usr/bin/env bash
# Finetune on Multilingual Complex Wikidata Questions
#
#SBATCH --job-name gqa-c
#SBATCH --output=runs/R-%x.%j.out
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
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
export WANDB__SERVICE_WAIT=300

# # Experiment c: CrossGNN with Graphs (No warmup; no frozen)
# python -u train.py --finetune --config configs/gqa.yaml --config-profile gqa_c_xwxf \
#     --run-name gqa-c-xwxf --tune-lr

# Test
python -u eval.py --config configs/gqa.yaml --config-profile gqa_e_xwxf \
    --run-name test-gqa-c --model t5-gnn --checkpoint-path logs/gqa/hfy9x0f7/checkpoints/epoch=83-step=4452.ckpt