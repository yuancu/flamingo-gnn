#!/usr/bin/env bash
# Finetune on Multilingual Complex Wikidata Questions
#
#SBATCH --job-name gqa-f
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

# # Experiment f: CrossGNN with FrozenLM and Warmup (Warm up, freeze LM)
# python -u train.py --finetune --config configs/gqa.yaml --config-profile gqa_f_wf \
#     --run-name gqa-f --tune-lr

# Test
python -u eval.py --config configs/gqa.yaml --config-profile gqa_f_wf \
    --run-name test-gqa-f --model t5-gnn --checkpoint-path logs/gqa/hoz4z6dy/checkpoints/epoch=98-step=5247.ckpt