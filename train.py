"""Pretrain the T5-based encoder-decoder architectured dragon model.
"""
import os
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dataset.lmgnn import load_data as load_lmgnn_data
from dataset.mutiple_choice import load_data as load_multiple_choice_data
from lightning.lit_seq2seq import LitT5Seq2Seq
from lightning.lit_multiple_choice import LitT5Seq2SeqForMultipleChoice
from models.flamingo_t5 import FlamingoConfig, FlamingoT5Decoder
from utils.common import load_args
from utils.model_utils import construct_encoder


def main(args):
    # 0. Set seed
    # Setting all the random seeds to the same value.
    # This is important in a distributed training setting. 
    # Each rank will get its own set of initial weights. 
    # If they don't match up, the gradients will not match either,
    # leading to training that may not converge.
    pl.seed_everything(1)

    # 1. Load configs
    run_name = args.run_name
    mode = 'pretrain' if args.pretrain else 'finetune'
    config_profile = args.config_profile
    multiple_choice = args.multiple_choice
    args = load_args(config_path=args.config, profile=args.config_profile)
    args.run_name = run_name

    # 2. Load data
    # Set collator and dataset according to the task: pretrain is mainly devided into two types:
    # with or without graph
    dummy_graph = hasattr(args, 'no_graph') and args.no_graph
    train_kwargs={'encoder_input': args.encoder_input, 'decoder_label': args.decoder_label}
    val_kwargs={'encoder_input': args.encoder_input, 'decoder_label': args.decoder_label}
    if mode == 'finetune' and not multiple_choice:
        val_kwargs['decoder_label'] = 'raw_answers'

    train_loader, val_loader = load_multiple_choice_data(
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
        dim_media=args.gnn_dim,
        xattn_dim_head=args.xattn_dim_head,
        xattn_heads=args.xattn_heads,
        xattn_every=args.xattn_every,
        xattn_ff_mult=args.xattn_ff_mult,
        lm_name_or_path=args.encoder_name_or_path,)
    decoder = FlamingoT5Decoder(decoder_config, encoder.get_input_embeddings())

    # 4. Create pytorch lightning model
    model_cls = LitT5Seq2SeqForMultipleChoice if multiple_choice else LitT5Seq2Seq
    if args.checkpoint_path and not args.restore_training: # if restore, we don't need to load the checkpoint here
        model = model_cls.load_from_checkpoint(
            args.checkpoint_path, strict=False,
            args=args, encoder=encoder, decoder=decoder,
            freeze_lm=args.freeze_lm, freeze_non_lm=args.freeze_non_lm,
            mode=mode
        )
    else:
        model = model_cls(
            args=args,encoder=encoder, decoder=decoder,
            freeze_lm=args.freeze_lm, freeze_non_lm=args.freeze_non_lm,
            mode=mode
        )

    # 5. Create trainer
    now = datetime.now().strftime('%m%d%H%M')
    run_name = f"{args.run_name}-{now}"
    offline = args.wandb_mode in ['offline', 'disabled']
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    wandb_logger = WandbLogger(project=args.wandb_project, offline=offline, name=run_name,
                               group=config_profile, save_dir=args.log_dir)
    # wandb_logger.experiment.config.update(vars(args))
    if mode == 'finetune':
        checkpoint_callback = ModelCheckpoint(monitor="em", mode="max", save_weights_only=True,)
        callbacks = [checkpoint_callback]
    else:
        checkpoint_callback = ModelCheckpoint(monitor="loss", mode="min", save_weights_only=True,
                                              dirpath="artifacts/pretrained", filename=args.run_name+".ckpt")
        callbacks = None
    trainer = pl.Trainer(max_epochs=args.n_epochs, fast_dev_run=args.fast_dev_run,
                         default_root_dir=os.path.join(args.save_dir, args.run_name),
                         accelerator='gpu', strategy=args.strategy, logger=wandb_logger,
                         callbacks=callbacks, gradient_clip_val=0.5,
                         accumulate_grad_batches=8)

    # 6. Train
    if args.restore_training:
        resume_checkpoint = args.checkpoint_path
        assert resume_checkpoint, "No checkpoint to resume training. (got {resume_checkpoint})"
    else:
        resume_checkpoint = None
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=resume_checkpoint)


if __name__ == '__main__':
    # To properly utilize a CUDA device with tensor cores
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.benchmark = True
    # To avoid 'too many open files' error
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/lmgnn.yaml')
    parser.add_argument('--config-profile', type=str, required=True)
    parser.add_argument('--run-name', type=str, required=True)
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--multiple-choice', action='store_true')
    args = parser.parse_args()
    if not args.pretrain ^ args.finetune:
        raise ValueError('Either pretrain or finetune should be set.')
    main(args)
