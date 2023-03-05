"""Pretrain the T5-based encoder-decoder architectured dragon model.
"""
import os
from functools import partial
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from dataset.dragon import DragonDataset, dragon_collate_fn, load_data
from lightning.lit_seq2seq import LitT5Seq2Seq
from models.t5_lmgnn import T5DragonEncoder
from models.flamingo_t5 import FlamingoT5Decoder, FlamingoConfig
from utils.common import load_args
from utils.model_utils import construct_encoder


def main(args):
    # 1. Load configs
    run_name = args.run_name
    config_profile = args.config_profile
    args = load_args(config_path=args.config, profile=args.config_profile)
    args.run_name = run_name

    # 2. Load data
    # Set collator and dataset according to the task: pretrain is mainly devided into two types:
    # with or without graph
    if hasattr(args, 'no_graph') and args.no_graph:
        collate_fn = partial(dragon_collate_fn, dummy_graph=True)
    else:
        collate_fn = partial(dragon_collate_fn, dummy_graph=False)
    train_loader, dev_loader, _ = load_data(
        args,
        dataset_cls=DragonDataset,
        collate_fn=collate_fn,
        corrupt=False,
        num_workers=8,
        dataset_kwargs={
            'encoder_input': 'context_prefix',
            'decoder_label': 'context_suffix',
            'prefix_ratio': 0.4})

    # 3. Create encoder and decoder
    encoder = construct_encoder(args, model_cls=T5DragonEncoder)
    # TODO: add a config file for flamingo
    decoder_config = FlamingoConfig(
        d_model=encoder.config.d_model,
        dim_media=100,
        xattn_dim_head=64,
        xattn_heads=8,
        xattn_every=1,
        xattn_ff_mult=4,
        lm_name_or_path=args.encoder_name_or_path,)
    decoder = FlamingoT5Decoder(decoder_config, encoder.get_input_embeddings())

    # 4. Create pytorch lightning model
    model = LitT5Seq2Seq(args=args,encoder=encoder, decoder=decoder,
                         freeze_encoder=True, freeze_decoder=True,
                         do_validation=False)

    # 5. Create trainer
    wandb_logger = WandbLogger(project=args.wandb_project, offline=True, name=args.run_name,
                               group=config_profile, save_dir=args.log_dir)
    wandb_logger.experiment.config.update(vars(args))
    trainer = pl.Trainer(max_epochs=args.n_epochs, fast_dev_run=args.fast_dev_run,
                         default_root_dir=os.path.join(args.save_dir, args.run_name),
                         gpus=1, logger=wandb_logger)

    # 6. Train
    if hasattr(args, 'resume_ckpt') and args.resume_ckpt:
        resume_ckpt = args.resume_ckpt
    else:
        resume_ckpt = None
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=dev_loader,
                ckpt_path=resume_ckpt)


if __name__ == '__main__':
    # To properly utilize a CUDA device with tensor cores
    torch.set_float32_matmul_precision('medium')

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/lmgnn.yaml')
    parser.add_argument('--config-profile', type=str, default='pretrain_squad')
    parser.add_argument('--run-name', type=str, required=True)
    args = parser.parse_args()
    main(args)
