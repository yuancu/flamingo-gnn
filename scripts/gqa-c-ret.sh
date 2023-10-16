#!/usr/bin/env bash
# Finetune on Multilingual Complex Wikidata Questions
#
#SBATCH --job-name gqa-cret
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

# Experiment c: CrossGNN with Graphs (No warmup; no frozen)
# python -u train.py --finetune --config configs/gqa.yaml --config-profile gqaret_c_xwxf \
#     --run-name gqa-c-xwxf-ret --tune-lr

# Test
python -u eval.py --config configs/gqa.yaml --config-profile gqaret_c_xwxf \
    --run-name test-gqa-c-ret --model t5-gnn --checkpoint-path logs/gqa/pfaqysh1/checkpoints/epoch=115-step=12296.ckpt