#!/usr/bin/env bash
export TOKENIZERS_PARALLELISM=true
export WANDB__SERVICE_WAIT=300

# run pretrain
python -u train.py --finetune --multiple-choice --config configs/obqa.yaml --config-profile finetune_obqa_sanity --run-name ftobqa-sanity