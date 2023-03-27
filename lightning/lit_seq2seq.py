import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim import AdamW, RMSprop
from transformers import AutoTokenizer
from transformers.optimization import Adafactor

from models.t5_seq2seq import T5Seq2Seq
from evaluation.squad import compute_score



def evaluate(predictions, references):
    score = compute_score(predictions, references)
    return score


class LitT5Seq2Seq(pl.LightningModule):
    def __init__(self, args, encoder, decoder, freeze_lm=True, freeze_non_lm=False,
                 do_validation=False, return_val_predictions=False):
        """
        Warning: the decoder_start_token_id will be initialized as the pad_token_id of a
        tokenizer constructed from args.encoder_name_or_path tokenizer.
        """
        super().__init__()
        self.validation_step_outputs = []
        self.save_hyperparameters(args)
        self.args = args
        # Freeze node embedding (duplicated but important)
        for n, p in encoder.named_parameters():
            if n.endswith("node_emb.emb.weight"):
                p.requires_grad = False
        # Freeze loaded weights from T5, GNN and XATTN is not frozen
        if freeze_lm:
            encoder.freeze_lm()
            decoder.freeze_lm()
        # Freeze the added parts (GNN & XATTN), T5 is not frozen
        if freeze_non_lm:
            encoder.freeze_non_lm()
            decoder.freeze_non_lm()
        # Construct a encoder-decoder model
        model = T5Seq2Seq(encoder=encoder, decoder=decoder)
        self.model = model
        # The tokenizer is used in the validation step
        tokenizer = AutoTokenizer.from_pretrained(args.encoder_name_or_path)
        self.tokenizer = tokenizer
        self.evaluator = evaluate
        # The current setting only allows validation in downstream tasks
        self.do_validation = do_validation
        self.return_val_predictions = return_val_predictions

    def training_step(self, batch, batch_idx):
        # input_ids, attention_mask, token_type_ids, special_token_mask, \
        #     decoder_labels,\
        #     node_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask,\
        #     edge_index, edge_type, pos_triples, neg_nodes = batch
        input_ids, attention_mask, decoder_labels, \
            node_ids, node_type_ids, adj_lengths, \
            edge_index, edge_type = batch
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
        input_ids, attention_mask, answers, \
            node_ids, node_type_ids, adj_lengths, \
            edge_index, edge_type = batch

        tokenizer = self.tokenizer
        gold_answers = answers
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
        self.validation_step_outputs.append(scores)
        return scores

    def on_validation_epoch_end(self):
        if not self.do_validation:
            return {}
        outputs = self.validation_step_outputs
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
        self.validation_step_outputs.clear()
        return scores

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def configure_optimizers(self):
        """Create an optimizer for the model, optionally using different learning rates for different layers.
        If use_ddp is True, the optimizer will be wrapped by DistributedDataParallel.
        """
        parameters = self.model.parameters()
        learning_rate = float(self.args.learning_rate)
        if self.args.optimizer == "adamw":
            optimizer = AdamW(parameters, lr=learning_rate)
        elif self.args.optimizer == "adafactor":
            # Set according to https://discuss.huggingface.co/t/t5-finetuning-tips/684/3
            optimizer = Adafactor(parameters, lr=learning_rate, scale_parameter=False,
                                  clip_threshold=1.0, relative_step=False)
        elif self.args.optimizer == "rmsprop":
            optimizer = RMSprop(parameters, lr=learning_rate)
        else:
            raise NotImplementedError(f"Optimizer {self.args.optimizer} is not supported.")
        # Load the optimizer state if resuming
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        # After moving the embedding to the CPU, this weight should no longer exist
        if 'model.encoder.node_emb.emb.weight' in checkpoint['state_dict']:
            del checkpoint['state_dict']['model.encoder.node_emb.emb.weight']
