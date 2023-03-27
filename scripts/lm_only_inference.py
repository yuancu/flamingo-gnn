"""Direct inference with pretrained LM-only model from huggingface hub."""
import argparse

import torch
from transformers import T5ForConditionalGeneration
from pytorch_lightning import Trainer
from torchsummary import summary

from lightning.lit_inference import LitSeq2SeqLMOnly
from utils.common import load_args
from dataset.lmgnn import load_data


def main(args):
    """
    args is expected to have the following fields:
        - model_name_or_path: str
            for loading pretrained model
        - profile: str
            for loading dataset
    """
    torch.set_float32_matmul_precision('medium')
    flan_t5 = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    lit_flan_t5 = LitSeq2SeqLMOnly(flan_t5)

    args = load_args(config_path=args.config_path, profile=args.profile)
    _, val_loader = load_data(
            args,
            corrupt=False,
            dummy_graph=True,
            num_workers=8,
            train_kwargs={
                'encoder_input': args.encoder_input,
                'decoder_label': args.decoder_label},
            val_kwargs={
                'encoder_input': args.encoder_input,
                'decoder_label': 'raw_answers'},)

    trainer = Trainer(accelerator='gpu', max_epochs=1)
    print("Model summary", summary(lit_flan_t5.model, depth=1))

    res = trainer.test(lit_flan_t5, dataloaders=val_loader)
    print(res)

if __name__ == '__main__':
    # Create the parser with fields config_path, profile and model_name_or_path
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/lmgnn.yaml')
    parser.add_argument('--profile', type=str, default='finetune_mcwq')
    parser.add_argument('--model_name_or_path', type=str, default='google/flan-t5-xxl')

    args = parser.parse_args()
    main(args)
