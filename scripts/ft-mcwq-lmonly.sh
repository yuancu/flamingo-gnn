#!/usr/bin/env bash

#SBATCH --job-name ftlm-mcwq
#SBATCH --output=runs/R-%x.%j.out
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1


source ~/miniconda3/bin/activate dragon

export TOKENIZERS_PARALLELISM=true

# run pretrain
python -u train_lm.py --config configs/mcwq.yaml --config-profile finetune_mcwq_lmonly_filtered --num-trainable-blocks -1 --run-name ft-mcwq-lmonly-filtered-adapter --adapter pfeiffer
