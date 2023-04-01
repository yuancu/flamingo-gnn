"""3. Filter question related facts

e.g.
python subgraph/filter_facts.py --statement-path data/squad/statement/ft.train.statement.jsonl \
    --fact-path data/squad/facts/ft.train.jsonl --max-triplet 100 --score-cache-path data/squad/scores/ft.train.pkl \
    --output-path data/squad/filtered_facts/ft.train.jsonl
"""
import argparse
import pickle
from pathlib import Path

import srsly
import torch
import torch.nn as nn
import truecase
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

FACT_FILTER_MODEL_PATH = 'artifacts/exaqt/fact_filter_model.bin'
WIKIDATA_PROP_DICTIONARY_PATH = 'artifacts/exaqt/wikidata_property_dictionary.json'


class FactFilterModel(nn.Module):
    """Fact filter binary classification model built on top of BERT.
    """
    def __init__(self, bert_path='bert-base-cased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids, **kwargs):
        outs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, **kwargs)
        bert_output = self.bert_drop(outs.pooler_output)
        output = self.out(bert_output)
        return output


def verbalize_triplet(triplet):
    # Keep only the lables of the triplets
    triplet = [e['label'] for e in triplet]
    sub, rel, obj = triplet[:3]

    quantifiers = triplet[:3]
    # Quantifiers appears in pairs, we group them together. e.g. point in time: 2023
    quantifiers = [' '.join(quantifiers[i:i+2]) for i in range(0, len(quantifiers), 2)]
    # In Wikidata, most properties are “has”-kind properties
    # ref: https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial
    verbalized_quantifiers = ' and has '.join(quantifiers)

    return sub + ' has ' + rel + ' ' + obj + verbalized_quantifiers


def score_triplets(triplets, question, model, tokenizer, batch_size=64):
    """Score a list of triplets in batch using the fact filter model.
    """
    verbalized_triplets = [verbalize_triplet(triplet) for triplet in triplets]
    scores = []
    for i in range(0, len(triplets), batch_size):
        tokenized_text = tokenizer.batch_encode_plus(
            [[question, verbalized_triplet] for verbalized_triplet in verbalized_triplets[i:i+batch_size]],
            padding=True, truncation=True, max_length=384, return_tensors='pt')
        tokenized_text = {k: v.to(next(model.parameters()).device) for k, v in tokenized_text.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            output = model(**tokenized_text)
        triplet_scores = torch.sigmoid(output).detach().cpu().numpy()
        triplet_scores = triplet_scores.flatten().tolist()
        assert len(triplet_scores) == len(triplets[i:i+batch_size])
        scores.extend(triplet_scores)
    return scores


def main(args):
    model = FactFilterModel()
    state_dict = torch.load(args.filter_model_path, map_location=torch.device('cpu'))
    # Remove 'module.' from the keys
    state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f'Fact filter model loaded from {args.filter_model_path}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    cnt_statement = sum(1 for _ in srsly.read_jsonl(args.statement_path))
    statements = srsly.read_jsonl(args.statement_path)
    facts = srsly.read_jsonl(args.fact_path)
    # Clear the output file
    with open(args.output_path, 'w', encoding='utf-8') as f:
        f.write('')

    # Scores is a dict of {id: [score1, score2, ...]}
    scores = {}
    score_loaded = False
    if args.score_cache_path is not None and Path(args.score_cache_path).exists():
        # Read the scores if exists && the length matches
        with open(args.score_cache_path, 'rb') as f:
            scores = pickle.load(f)
            if len(scores) == cnt_statement:
                print(f'Loading scores from {args.score_cache_path}')
                scores = srsly.read_jsonl(args.score_cache_path)
                score_loaded = True
            else:
                print(f'Length of scores ({len(scores)}) does not match the length of statements ({cnt_statement}). Recomputing scores.')

    for fact, statement in tqdm(zip(facts, statements), total=cnt_statement, desc='Filtering facts'):
        assert fact['id'] == statement['id']
        # e.g. new york -> New York
        question = truecase.get_true_case(statement['question'])
        triplets = fact['facts']
        if fact['id'] in scores:
            triplet_scores = scores[fact['id']]
        else:
            triplet_scores = score_triplets(triplets, question, model, tokenizer, batch_size=args.batch_size)
            scores[fact['id']] = triplet_scores
        # Sort triplets by their scores and keep the index
        sorted_scores = sorted(zip(triplet_scores, range(len(triplet_scores))), reverse=True)
        # Keep the top-k triplets
        filtered_triplets = [{'score': score, 'triplet': triplets[idx]} for score, idx in sorted_scores[:args.max_triplet]]
        filtered_fact = {
            'id': fact['id'],
            'question': statement['question'],
            'facts': filtered_triplets
        }
        # Write the filtered top-k facts for each question
        srsly.write_jsonl(args.output_path, [filtered_fact], append=True)
    
    # Save the scores if score_cache_path is specified ad the scores are not loaded
    if args.score_cache_path is not None and not score_loaded:
        # Save the scores
        with open(args.score_cache_path, 'wb') as f:
            pickle.dump(scores, f)
            print(f'Scores saved to {args.score_cache_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter-model-path', type=str, default=FACT_FILTER_MODEL_PATH, help='Path to the fact filter model')
    parser.add_argument('--wikidata-prop-dict-path', type=str, default=WIKIDATA_PROP_DICTIONARY_PATH, help='Path to the Wikidata property dictionary')
    parser.add_argument('--batch-size', type=int, default=196, help='Inference batch size when scoring the facts')
    parser.add_argument('--statement-path', type=str, required=True, help='Path to the statements file, which contains the questions')
    parser.add_argument('--fact-path', type=str, required=True, help='Path to the facts file, which contains the facts related to the questions')
    parser.add_argument('--max-triplet', type=int, default=1000, help='Maximum number of triplets to be kept')
    parser.add_argument('--score-cache-path', type=str, default=None, help='Path to store the score cache file. If omitted, the scores of all facts will not be saved.')
    parser.add_argument('--output-path', type=str, required=True, help='Path to the score file')
    args = parser.parse_args()

    main(args)

