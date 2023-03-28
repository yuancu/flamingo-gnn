"""
This module define inference procedure for LM-only models.
"""
import torch
import numpy as np
import pytorch_lightning as pl
from transformers import AutoTokenizer

from evaluation.squad import compute_score


def evaluate(predictions, references):
    score = compute_score(predictions, references)
    return score

class LitSeq2SeqLMOnly(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.test_step_outputs = []
        self.model = model
        if hasattr(model, 'name_or_path'):
            self.name_or_path = model.name_or_path
        elif hasattr(model, 'config') and hasattr(model.config, 'name_or_path'):
            self.name_or_path = model.config.name_or_path
        else:
            raise ValueError('Cannot find the name_or_path of the model.')
        self.tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        self.evaluator = evaluate

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, answers, \
            node_ids, node_type_ids, adj_lengths, \
            edge_index, edge_type = batch

        tokenizer = self.tokenizer
        gold_answers = answers
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask)
        predictions = tokenizer.batch_decode(generated)
        predictions = [p.replace(tokenizer.pad_token, '').replace(tokenizer.eos_token, '').strip() for p in predictions]
        scores = self.evaluator(predictions, gold_answers)
        self.log_dict(scores)
        self.test_step_outputs.append(scores)
        return scores

    def on_test_step_end(self):
        outputs = self.test_step_outputs
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
        self.test_step_outputs.clear()
        return scores

    def predict_step(self, batch, batch_idx) :
        return self.test_step(batch, batch_idx)
