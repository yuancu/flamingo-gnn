#!/usr/bin/env bash
#
#SBATCH --job-name gqa-langemb
#SBATCH --output=runs/R-%x.%j.out
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1


source ~/miniconda3/bin/activate dragon

export TOKENIZERS_PARALLELISM=true
export WANDB__SERVICE_WAIT=300

python -u train.py --finetune --config configs/gqa.yaml --config-profile gqa_langemb --run-name ft-gqa-lang --tune-lr