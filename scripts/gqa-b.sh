#!/usr/bin/env bash
# Finetune on Graph Question Answering (GQA) dataset
#
#SBATCH --job-name gqa-lmctx
#SBATCH --output=runs/R-%x.%j.out
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
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

# unfrozen
# python -u train_lm.py --config configs/gqa.yaml --config-profile gqa_b_lmctx \
#     --run-name gqa-b  --num-trainable-blocks -1

# Test
python -u eval.py --config configs/gqa.yaml --config-profile gqa_b_lmctx \
    --run-name test-gqa-b --model t5 --checkpoint-path logs/gqa/mhxl78fv/checkpoints/epoch=1-step=106.ckpt