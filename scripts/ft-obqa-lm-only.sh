#!/usr/bin/env bash
# Finetune on Multilingual Complex Wikidata Questionsn with dummy graph
#
#SBATCH --job-name ftcwqng
#SBATCH --output=runs/R-%x.%j.out
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

# debug info
hostname
which python
nvidia-smi

# venv
# source ~/miniconda3/bin/activate dragon

export TOKENIZERS_PARALLELISM=true

# run pretrain
python -u finetune_t5.py --multiple-choice --config configs/obqa.yaml --config-profile finetune_obqa_no_graph --num-trainable-blocks 6 --run-name ft-obqa-lm-only-trainable6