#!/usr/bin/env bash
# Finetune on Multilingual Complex Wikidata Questions with Adapter
#
#SBATCH --job-name ftadapter-mcwq
#SBATCH --output=runs/R-%x.%j.out
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

source ~/miniconda3/bin/activate dragon

export TOKENIZERS_PARALLELISM=true
export WANDB__SERVICE_WAIT=300

python -u train.py --finetune --add-adapter --config configs/mcwq.yaml --config-profile finetune_mcwq --run-name ft-mcwq-adapter