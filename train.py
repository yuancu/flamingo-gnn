"""Pretrain the T5-based encoder-decoder architectured dragon model.
"""
import os
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from dataset.lmgnn import load_data
from lightning.lit_seq2seq import LitT5Seq2Seq
from models.flamingo_t5 import FlamingoT5Decoder, FlamingoConfig
from utils.common import load_args
from utils.model_utils import construct_encoder


def main(args):
    # 1. Load configs
    run_name = args.run_name
    mode = 'pretrain' if args.pretrain else 'finetune'
    config_profile = args.config_profile
    args = load_args(config_path=args.config, profile=args.config_profile)
    args.run_name = run_name

    # 2. Load data
    # Set collator and dataset according to the task: pretrain is mainly devided into two types:
    # with or without graph
    if hasattr(args, 'no_graph') and args.no_graph:
        dummy_graph = True
    else:
        dummy_graph = False
    train_kwargs={'encoder_input': 'contextualized_question', 'decoder_label': 'answer'}
    val_kwargs={'encoder_input': 'contextualized_question', 'decoder_label': 'raw_answers'}
    train_loader, val_loader = load_data(
        args,
        corrupt=False,
        dummy_graph=dummy_graph,
        num_workers=8,
        train_kwargs=train_kwargs,
        val_kwargs=val_kwargs,)

    # 3. Create encoder and decoder
    encoder = construct_encoder(args)
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
                         do_validation=(mode=='finetune'))

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
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=resume_ckpt)


if __name__ == '__main__':
    # To properly utilize a CUDA device with tensor cores
    torch.set_float32_matmul_precision('medium')

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/lmgnn.yaml')
    parser.add_argument('--config-profile', type=str, required=True)
    parser.add_argument('--run-name', type=str, required=True)
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    args = parser.parse_args()
    if not args.pretrain ^ args.finetune:
        raise ValueError('Either pretrain or finetune should be set.')
    main(args)
