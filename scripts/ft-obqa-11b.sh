#!/usr/bin/env bash
export TOKENIZERS_PARALLELISM=true

# run pretrain
python -u finetune_t5.py --multiple-choice --config configs/obqa.yaml --config-profile finetune_obqa_11b_adapter --num-trainable-blocks -1 --run-name ft-obqa-t511b-lora --adapter lora