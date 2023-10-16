#!/usr/bin/env bash
# Finetune on Multilingual Complex Wikidata Questions
#
#SBATCH --job-name gqa-dret
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

# Experiment d: CrossGNN with Warmup (Warm up, No frozen)
# python -u train.py --finetune --config configs/gqa.yaml --config-profile gqaret_d_xf \
#     --run-name gqa-d-xf-ret --tune-lr

# Test
python -u eval.py --config configs/gqa.yaml --config-profile gqaret_d_xf \
    --run-name test-gqa-d-ret --model t5-gnn --checkpoint-path logs/gqa/4immse5f/checkpoints/epoch=61-step=4402.ckpt