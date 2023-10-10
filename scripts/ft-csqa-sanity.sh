#!/usr/bin/env bash
export TOKENIZERS_PARALLELISM=true
export WANDB__SERVICE_WAIT=300

# run pretrain
python -u train.py --finetune --multiple-choice --config configs/csqa.yaml --config-profile finetune_csqa_sanity --run-name ftcsqa-sanity