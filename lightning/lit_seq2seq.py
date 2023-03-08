import numpy as np
import pytorch_lightning as pl
import torch
from evaluate import load
from torch.optim import AdamW
from transformers import AutoTokenizer

from models.t5_seq2seq import T5Seq2Seq
from utils.model_utils import sep_params
from evaluation.squad import compute_score


def create_evaluator():
    """It returns a evaluate function that computes rouge and exact match scores."""
    def evaluate(predictions, references):
        score = compute_score(predictions, references)
        return score
    return evaluate


class LitT5Seq2Seq(pl.LightningModule):
    def __init__(self, args, encoder, decoder, freeze_encoder=True, freeze_decoder=True,
                 do_validation=False, return_val_predictions=False):
        """
        Warning: the decoder_start_token_id will be initialized as the pad_token_id of a
        tokenizer constructed from args.encoder_name_or_path tokenizer.
        """
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args
        # Freeze node embedding (duplicated but important)
        for n, p in encoder.named_parameters():
            if n.endswith("node_emb.emb.weight"):
                p.requires_grad = False
        # Freeze loaded weights from T5, GNN part is not frozen
        if freeze_encoder:
            encoder.freeze_lm()
        # Freeze decoder
        if freeze_decoder:
            decoder.freeze_lm()
        # Construct a encoder-decoder model
        model = T5Seq2Seq(encoder=encoder, decoder=decoder)
        self.model = model
        # The tokenizer is used in the validation step
        tokenizer = AutoTokenizer.from_pretrained(args.encoder_name_or_path)
        self.tokenizer = tokenizer
        self.evaluator = create_evaluator()
        # The current setting only allows validation in downstream tasks
        self.do_validation = do_validation
        self.return_val_predictions = return_val_predictions

    def training_step(self, batch, batch_idx):
        # input_ids, attention_mask, token_type_ids, special_token_mask, \
        #     decoder_labels,\
        #     node_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask,\
        #     edge_index, edge_type, pos_triples, neg_nodes = batch
        input_ids, attention_mask, token_type_ids, special_token_mask, decoder_labels, \
        node_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask, \
        edge_index, edge_type, pos_triples, neg_nodes = batch
        # for debugging
        assert attention_mask.shape == input_ids.shape
        # lm_input_ids as inputs, input_ids as labels, here they share the same attention mask
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            node_ids=node_ids,
            node_type_ids=node_type_ids,
            adj_lengths=adj_lengths,
            edge_index=edge_index,
            edge_type=edge_type,
            output_attentions=True,
            output_hidden_states=True,
            labels=decoder_labels,
            return_dict=True)

        loss = output.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if not self.do_validation:
            return {}
        input_ids, attention_mask, token_type_ids, special_token_mask, decoder_labels, \
        node_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask, \
        edge_index, edge_type, pos_triples, neg_nodes = batch

        tokenizer = self.tokenizer
        # replace ignore index with pad token id
        decoder_labels[decoder_labels==-100] = tokenizer.pad_token_id
        gold_answers = tokenizer.batch_decode(decoder_labels)
        gold_answers = [a.replace(tokenizer.pad_token, '').replace(tokenizer.eos_token, '').strip() for a in gold_answers]
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                node_ids=node_ids,
                node_type_ids=node_type_ids,
                adj_lengths=adj_lengths,
                edge_index=edge_index,
                edge_type=edge_type,)
        predictions = tokenizer.batch_decode(generated)
        predictions = [p.replace(tokenizer.pad_token, '').replace(tokenizer.eos_token, '').strip() for p in predictions]
        scores = self.evaluator(predictions, gold_answers)
        self.log_dict(scores)
        if self.return_val_predictions:
            return {
                "predictions": predictions,
                "references": gold_answers,
                **scores
            }
        return scores

    def validation_epoch_end(self, outputs):
        if not self.do_validation:
            return {}
        if len(outputs) > 0:
            mean_keys = outputs[0].keys()
            mean_keys = set(mean_keys) - set(["predictions", "references"])
            scores = {k: np.mean([o[k] for o in outputs]) for k in mean_keys}
            if "predictions" in outputs[0]:
                # concatenate all predictions
                scores["predictions"] = [p for o in outputs for p in o["predictions"]]
            if "references" in outputs[0]:
                # concatenate all references
                scores["references"] = [p for o in outputs for p in o["references"]]
        else:
            scores = {}
        return scores

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def configure_optimizers(self):
        """Create an optimizer for the model, optionally using different learning rates for different layers.
        If use_ddp is True, the optimizer will be wrapped by DistributedDataParallel.
        """
        if self.args.different_lr:
            loading_info = self.model.encoder.loading_info
            loaded_params, not_loaded_params = sep_params(self.model, loading_info, prefix="lmgnn.")
            small_lr_params, large_lr_params = loaded_params, not_loaded_params
            if self.args.use_ddp:
                assert next(iter(small_lr_params.keys())).startswith('module.'), \
                    "The small_lr_params should be updated by DDP."
                assert next(iter(large_lr_params.keys())).startswith('module.'), \
                    "The large_lr_params should be updated by DDP."
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            small_lr = float(self.args.small_lr)
            large_lr = float(self.args.large_lr)
            parameters = [
                {'params': [p for n, p in small_lr_params.items() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay, 'lr': small_lr},
                {'params': [p for n, p in small_lr_params.items() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': small_lr},
                {'params': [p for n, p in large_lr_params.items() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay, 'lr': large_lr},
                {'params': [p for n, p in large_lr_params.items() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': large_lr},
            ]
            optimizer = AdamW(parameters)
        else:
            parameters = self.model.parameters()
            learning_rate = float(self.args.large_lr)
            optimizer = AdamW(parameters, lr=learning_rate)
        # Load the optimizer state if resuming
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        del checkpoint['state_dict']['model.encoder.lmgnn.node_emb.emb.weight']
