import json
import logging
import math
import os
import pickle
import random
from abc import ABC, abstractmethod
from collections import namedtuple
from functools import lru_cache, partial
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AutoTokenizer, BertTokenizer, BertTokenizerFast,
                          RobertaTokenizer, RobertaTokenizerFast, T5Tokenizer,
                          T5TokenizerFast)
from transformers.data import DataCollatorForSeq2Seq

from utils.model_utils import get_tweaked_num_relations

InputExample = namedtuple('InputExample', 'example_id contexts question endings label')
InputFeatures = namedtuple('InputFeatures', 'example_id choices_features label')


class DragonDataset(Dataset):
    """This dataset additionally outputs decoder inputs on top of DragonDataset with 
    dragond_enc_collate_fn's output. It removes the annoying choices dimension. Besides,
    it doesn't corrupt the graph and the text for the link prediction task and masked
    language modeling task.
    """
    def __init__(self, statement_path, num_relations, adj_path, legacy_adj,
                 max_seq_length=256, model_name='t5-base', max_node_num=200,
                 cxt_node_connects_all=False,kg_only_use_qa_nodes=False,
                 num_choices=1, link_drop_max_count=100, link_drop_probability=0.2,
                 link_drop_probability_in_which_keep=0.2, link_negative_sample_size=64,
                 corrupt_graph=False, corrupt_text=False, span_mask=False,
                 mlm_probability=0.15, truncation_side='right', encoder_input='question',
                 decoder_label='answer', prefix_ratio=0.5):
        """
        Args:
        adj_path: the path to a monilithic adj pickle (legacy) or path to a folder containing adj pickle files
        legacy_mode: if True, use the monolithic adj pickle file, else the adj_path should be a folder
        max_num_relation: the maximum number of kg relations to keep. (deprecated)
        encoder_input: the input to the encoder. Can be 'question', 'context', or 'contextualized_question'
        """
        # Valid pairs
        # - Pretraining (MLM): encoder_input = 'context', decoder_label = 'context_label' | 'context'
        # - Pretraining (Completion): encoder_input = 'context_prefix', decoder_label = 'context_suffix'
        # - Finetuning: encoder_input = 'question' | 'retrieval_augmented_question', decoder_label = 'answer'
                # assert is file is legacy_mode; else assert it is a folder
        assert encoder_input in ['question', 'context', 'context_prefix', 'retrieval_augmented_question']
        assert decoder_label  in ['answer', 'context_label',  'context', 'context_suffix']
        if decoder_label == 'context_label':
            assert corrupt_text, "corrupt_text must be True for MLM tasks (when decoder_label='context_label')"
        if encoder_input == 'context_prefix' or decoder_label == 'context_suffix':
            assert encoder_input == 'context_prefix' and decoder_label == 'context_suffix', \
                "'context_prefix' and 'context_suffix' must be used together'"
        if legacy_adj:
            assert os.path.isfile(adj_path), "adj_path should be a file in legacy mode"
        else:
            assert os.path.isdir(adj_path), "adj_path should be a folder in non-legacy mode"
        if corrupt_graph and (encoder_input == 'question' or encoder_input == 'retrieval_augmented_question'):
            logging.warning("corrupt_graph shouldn't be set for downstream tasks. Setting it to False.")
            corrupt_graph = False

        super(Dataset).__init__()
        # For text data
        self.max_seq_length = max_seq_length
        # truncation_side is only available in newer versions of transformers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side=truncation_side)
        if isinstance(self.tokenizer, (T5TokenizerFast, T5Tokenizer)):
            self.tokenizer.sep_token = '<extra_id_1>'
            self.tokenizer.cls_token = '<extra_id_2>'
            self.tokenizer.mask_token = '<extra_id_3>'
        # Read statements
        self.examples = self.read_statements(statement_path)
        if len(self.examples) == 0:
            raise ValueError("No examples found in the dataset")
        self.num_choices = num_choices
        self.corrupt_text = corrupt_text
        self.span_mask = span_mask
        self.mlm_probability = mlm_probability
        # TODO: Explain the following parameters
        self.geo_p = 0.2
        self.span_len_upper = 10
        self.span_len_lower = 1
        self.span_lens = list(range(self.span_len_lower, self.span_len_upper + 1))
        self.span_len_dist = [self.geo_p * (1-self.geo_p)**(i - self.span_len_lower) for i in range(self.span_len_lower, self.span_len_upper + 1)]
        self.span_len_dist = [x / (sum(self.span_len_dist)) for x in self.span_len_dist]
        self.encoder_input = encoder_input
        self.decoder_label = decoder_label

        # For graph data
        self.legacy_adj = legacy_adj
        self.all_adjs= None  # only for legacy mode
        self.adj_path = adj_path
        self.cxt_node_connects_all = cxt_node_connects_all
        self.num_relations = num_relations
        self.max_node_num = max_node_num
        self.kg_only_use_qa_nodes = kg_only_use_qa_nodes
        self.link_drop_max_count = link_drop_max_count
        self.link_drop_probability = link_drop_probability
        self.link_drop_probability_in_which_keep = link_drop_probability_in_which_keep
        self.link_negative_sample_size = link_negative_sample_size
        self.corrupt_graph = corrupt_graph

        # For retrieval
        if encoder_input == 'retrieval_augmented_question':
            self.retriever = Retriever()

        # For prefix completion
        self.prefix_ratio = prefix_ratio

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # 1. Load text data
        # self.train_qids, self.train_labels, self.train_encoder_data, train_nodes_by_sents_list
        example = self.examples[idx]
        # MLM is also done in postprocess_text
        encoder_inputs, decoder_labels = self.postprocess_text(example)
        input_ids, attention_mask, token_type_ids, special_tokens_mask = encoder_inputs["input_ids"], encoder_inputs["attention_mask"], \
            encoder_inputs["token_type_ids"], encoder_inputs["special_tokens_mask"]

        # 2. Load graph data, post processing is done in the _load_graph_from_index function
        node_ids, node_type_ids, node_scores, adj_length, special_nodes_mask,\
            input_edge_index, input_edge_type, pos_triples, neg_nodes = self._load_graph_from_index(idx)

        return input_ids, attention_mask, token_type_ids, special_tokens_mask, decoder_labels,\
            node_ids, node_type_ids, node_scores, adj_length, special_nodes_mask,\
            input_edge_index, input_edge_type, pos_triples, neg_nodes

    def _load_graph_from_index(self, idx):
        """adapted from load_sparse_adj_data_with_contextnode in utils/data_utils.py L653"""
        if self.legacy_adj:
            graph = self.load_graph_legacy(idx)
        else:
            example = self.examples[idx]
            graph = self.load_graph(example.example_id)
        *decoder_data, adj_data = self.preprocess_graph(graph)
        node_ids, node_type_ids, node_scores, adj_length, special_nodes_mask = decoder_data
        edge_index, edge_type = adj_data

        input_edge_index, input_edge_type, pos_triples, neg_nodes = self.postprocess_graph(edge_index, edge_type, node_type_ids)
        return node_ids, node_type_ids, node_scores, adj_length, special_nodes_mask,\
            input_edge_index, input_edge_type, pos_triples, neg_nodes

    def load_graph(self, example_id):
        """
        Load subgraph from adj_folder/example_id.pkl 
        """
        graph_path = os.path.join(self.adj_path, example_id + ".pkl")
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
        return graph

    def load_graph_legacy(self, idx):
        """
        Directly load a subgraph from a pickled list.
        """
        if self.all_adjs is None:
            with open(self.adj_path, "rb") as f:
                self.all_adjs = pickle.load(f)
        return self.all_adjs[idx]

    def preprocess_graph(self, graph):
        """
        Adapted from load_sparse_adj_data_with_contextnode in utils/data_utils.py L653
        An example is a tuple of (adj, nodes, qmask, amask, statement_id)
        Returns:
            node_ids: [n_choice, max_node_num]
            node_type_ids: [n_choice, max_node_num]
            node_scores: [n_choice, max_node_num, 1]
            adj_length: (1,)
            special_nodes_mask: [n_choice, max_node_num]
            (edge_index, edge_type): [n_choice, 2, max_edge_num], [n_choice, max_edge_num]
        """
        # Define special nodes and links
        context_node = 0
        n_special_nodes = 1
        cxt2qlinked_rel = 0
        cxt2alinked_rel = 1
        half_n_rel = get_tweaked_num_relations(self.num_relations, self.cxt_node_connects_all)
        if self.cxt_node_connects_all:
            cxt2other_rel = half_n_rel - 1

        node_ids = torch.full((self.max_node_num,), 1, dtype=torch.long)
        node_type_ids = torch.full((self.max_node_num,), 2, dtype=torch.long) #default 2: "other node"
        node_scores = torch.zeros((self.max_node_num, 1), dtype=torch.float)
        special_nodes_mask = torch.zeros((self.max_node_num,), dtype=torch.bool)

        # adj: e.g. <4233x249 (n_nodes*half_n_rels x n_nodes) sparse matrix of type '<class 'numpy.bool'>' with 2905 stored elements in COOrdinate format>
        # nodes: np.array(num_nodes, ), where entry is node id
        # qm: np.array(num_nodes, ), where entry is True/False
        # am: np.array(num_nodes, ), where entry is True/False
        adj, nodes, qmask, amask = graph[:4]
        assert len(nodes) == len(set(nodes))

        qamask = qmask | amask
        # Sanity check: should be T,..,T,F,F,..F
        if len(nodes) == 0:
            pass
        else:
            # qamask[0] is np.bool_ type, is operator doesn't work
            assert qamask[0], "The existing nodes should not be masked"
        f_start = False
        for tf in qamask:
            if tf is False:
                f_start = True
            else:
                assert f_start is False

        assert n_special_nodes <= self.max_node_num
        special_nodes_mask[:n_special_nodes] = 1
        if self.kg_only_use_qa_nodes:
            actual_max_node_num = torch.tensor(qamask).long().sum().item()
        else:
            actual_max_node_num = self.max_node_num

        num_node = min(len(nodes) + n_special_nodes, actual_max_node_num) # this is the final number of nodes including contextnode but excluding PAD
        adj_lengths_ori = torch.tensor(len(nodes))
        adj_length = torch.tensor(num_node)

        # Prepare nodes
        nodes = nodes[:num_node - n_special_nodes]
        node_ids[n_special_nodes:num_node] = torch.tensor(nodes) + 1  # To accomodate contextnode, original node_ids incremented by 1
        node_ids[0] = context_node # this is the "node_id" for contextnode

        # Prepare node types
        node_type_ids[0] = 3 # context node
        node_type_ids[1:n_special_nodes] = 4 # sent nodes
        node_type_ids[n_special_nodes:num_node][torch.tensor(qmask, dtype=torch.bool)[:num_node - n_special_nodes]] = 0
        node_type_ids[n_special_nodes:num_node][torch.tensor(amask, dtype=torch.bool)[:num_node - n_special_nodes]] = 1

        #Load adj
        ij = torch.tensor(adj.row, dtype=torch.int64) #(num_matrix_entries, ), where each entry is coordinate
        k = torch.tensor(adj.col, dtype=torch.int64)  #(num_matrix_entries, ), where each entry is coordinate
        n_node = adj.shape[1]
        if n_node > 0:
            assert self.num_relations == adj.shape[0] // n_node
            i, j = torch.div(ij, n_node, rounding_mode='floor'), ij % n_node
        else:
            i, j = ij, ij

        #Prepare edges
        # **** increment coordinate by 1, rel_id by 2 ****
        i += 2
        j += 1
        k += 1
        extra_i, extra_j, extra_k = [], [], []
        for _coord, q_tf in enumerate(qmask):
            _new_coord = _coord + n_special_nodes
            if _new_coord > num_node:
                break
            if q_tf:
                extra_i.append(cxt2qlinked_rel) #rel from contextnode to question node
                extra_j.append(0) #contextnode coordinate
                extra_k.append(_new_coord) #question node coordinate
            elif self.cxt_node_connects_all:
                extra_i.append(cxt2other_rel) #rel from contextnode to other node
                extra_j.append(0) #contextnode coordinate
                extra_k.append(_new_coord) #other node coordinate
        for _coord, a_tf in enumerate(amask):
            _new_coord = _coord + n_special_nodes
            if _new_coord > num_node:
                break
            if a_tf:
                extra_i.append(cxt2alinked_rel) #rel from contextnode to answer node
                extra_j.append(0) #contextnode coordinate
                extra_k.append(_new_coord) #answer node coordinate
            elif self.cxt_node_connects_all:
                extra_i.append(cxt2other_rel) #rel from contextnode to other node
                extra_j.append(0) #contextnode coordinate
                extra_k.append(_new_coord) #other node coordinate

        # half_n_rel += 2 #should be 19 now
        if len(extra_i) > 0:
            i = torch.cat([i, torch.tensor(extra_i)], dim=0)
            j = torch.cat([j, torch.tensor(extra_j)], dim=0)
            k = torch.cat([k, torch.tensor(extra_k)], dim=0)

        ########################

        mask = (j < actual_max_node_num) & (k < actual_max_node_num)
        i, j, k = i[mask], j[mask], k[mask]
        i, j, k = torch.cat((i, i + half_n_rel), 0), torch.cat((j, k), 0), torch.cat((k, j), 0)  # add inverse relations
        edge_index = torch.stack([j,k], dim=0) # each entry is [2, E]
        edge_type = i # each entry is [E, ]
        edge_index = [edge_index] * self.num_choices
        edge_type = [edge_type] * self.num_choices

        adj_lengths_ori, node_ids, node_type_ids, node_scores, adj_length, special_nodes_mask = [x.view(self.num_choices, *x.size()) for x in (adj_lengths_ori, node_ids, node_type_ids, node_scores, adj_length, special_nodes_mask)]
        # node_ids: (n_choice, max_node_num)
        # node_type_ids: (n_choice, max_node_num)
        # node_scores: (n_choice, max_node_num)
        # adj_lengths: (n_choice)
        # edge_index: ([2, E]) * n_choice
        # edge_type: ([E, ]) * n_choice
        return node_ids, node_type_ids, node_scores, adj_length, special_nodes_mask, (edge_index, edge_type)

    def postprocess_graph(self, edge_index, edge_type, node_type_ids):
        """Adapted from data_utils.py L319: process_graph_data
        It corrupts the graph is self.corrupt_graph is True
        """
        # edge_index: list of shape (num_choice), where each entry is tensor[2, E]
        # edge_type: list of shape (num_choice), where each entry is tensor[E, ]
        # node_type_ids: tensor[num_choice, num_nodes]
        nc = len(edge_index)
        input_edge_index, input_edge_type, pos_triples, neg_nodes = [], [], [], []
        for cid in range(nc):
            _edge_index = edge_index[cid] #.clone()
            _edge_type  = edge_type[cid] #.clone()
            _node_type_ids = node_type_ids[cid] #.clone()
            _edge_index, _edge_type, _pos_triples, _neg_nodes = self.negative_sample_graph(_edge_index, _edge_type, _node_type_ids)
            input_edge_index.append(_edge_index)
            input_edge_type.append(_edge_type)
            pos_triples.append(_pos_triples)
            neg_nodes.append(_neg_nodes)

        if not self.corrupt_graph:
            input_edge_index = edge_index  #non-modified input
            input_edge_type = edge_type    #non-modified input

        return input_edge_index, input_edge_type, pos_triples, neg_nodes

    def negative_sample_graph(self, _edge_index, _edge_type, _node_type_ids):
        """Adapted from data_utils.py L346: _process_one_graph
        Args:
            _edge_index: tensor[2, E]
            _edge_type:  tensor[E, ]
            _node_type_ids: tensor[n_nodes, ]
        Returns:
            input_edge_index
            input_edge_type
            pos_triples: list: [h, r, t]
            neg_nodes
        """

        E = len(_edge_type)
        if E == 0:
            # print ('KG with 0 node', file=sys.stderr)
            effective_num_nodes = 1
        else:
            effective_num_nodes = int(_edge_index.max()) + 1
        device = _edge_type.device

        tmp = _node_type_ids.max().item()
        assert isinstance(tmp, int) and 0 <= tmp <= 5
        _edge_index_node_type = _node_type_ids[_edge_index] #[2, E]
        _is_special = (_edge_index_node_type == 3) #[2, E]
        is_special = _is_special[0] | _is_special[1] #[E,]

        positions = torch.arange(E)
        positions = positions[~is_special] #[some_E, ]
        drop_count = min(self.link_drop_max_count, int(len(positions) * self.link_drop_probability))
        if len(positions) > 0 and drop_count > 0:
            drop_idxs = torch.multinomial(torch.full((len(positions),), 1.0), drop_count, replacement=False) #[drop_count, ]
        else:
            drop_idxs = torch.tensor([]).long()
        drop_positions = positions[drop_idxs] #[drop_count, ]

        mask = torch.zeros((E,)).long() #[E, ]
        mask = mask.index_fill_(dim=0, index=drop_positions, value=1).bool().to(device) #[E, ]

        real_drop_count = int(drop_count * (1-self.link_drop_probability_in_which_keep))
        real_drop_positions = positions[drop_idxs[:real_drop_count]] #[real_drop_count, ]
        real_mask = torch.zeros((E,)).long() #[E, ]
        real_mask = real_mask.index_fill_(dim=0, index=real_drop_positions, value=1).bool().to(device) #[E, ]

        assert int(mask.long().sum()) == drop_count
        # print (f'drop_E / total_E = {drop_count} / {E} = {drop_count / E}', ) #E is typically 1000-3000
        input_edge_index = _edge_index[:, ~real_mask]
        input_edge_type  = _edge_type[~real_mask]
        assert input_edge_index.size(1) == E - real_drop_count

        pos_edge_index = _edge_index[:, mask]
        pos_edge_type  = _edge_type[mask]
        pos_triples = [pos_edge_index[0], pos_edge_type, pos_edge_index[1]]
        #pos_triples: list[h, r, t], where each of h, r, t is [n_triple, ]
        assert pos_edge_index.size(1) == drop_count

        num_edges = len(pos_edge_type)
        num_corruption = self.link_negative_sample_size
        neg_nodes = torch.randint(0, effective_num_nodes, (num_edges, num_corruption), device=device) #[n_triple, n_neg]
        return input_edge_index, input_edge_type, pos_triples, neg_nodes

    def read_statements(self, input_file):
        """
        Retruns:
            example_id (str): id
            contexts (str): context if exists, otherwise ""
            question (str): question
            endings (str): answer
            label (int): label, -100 in our case
        """
        with open(input_file, "r", encoding="utf-8") as f:
            examples = []
            for line in tqdm(f.readlines()):
                json_dic = json.loads(line)
                label = ord(json_dic["answerKey"]) - ord("A") if 'answerKey' in json_dic else -100
                question = json_dic["question"]["stem"]
                choices = json_dic["question"]["choices"]
                assert len(choices) > 0
                answer = choices[label]["text"] if label != -100 else choices[0]["text"]
                if "context" in json_dic:
                    context = json_dic["context"]
                else:
                    context = ""
                assert isinstance(context, str), "Currently only support string context"
                examples.append(
                    InputExample(
                        example_id=json_dic["id"],
                        contexts=context,
                        question=question,
                        endings=answer,
                        label=label
                    ))
        return examples

    def postprocess_text(self, example):
        """Adapted from load_input_tensors in utils/data_utils.py L584
        Args:
            example: an InputExample object
        Returns:
            example_id (str): the id of the example
            data_tensor (tensor [n_choice, max_seq_len] each): a tuple of input_ids, input_mask, segment_ids, output_mask.
            nodes_by_sents (list: empty): a list of nodes in the context
        """
        # There is no choice in out current setting, this is purely for compatibility with the original code

        context = example.contexts
        assert isinstance(context, str), "Currently contexts must be a string"
        question = example.question
        answer = example.endings
        # Here we separately encode the question and answer
        # Warning: we assume the encoder and the decoder share the same tokenizer
        # but this is not necessarily true!
        # Padding is removed here, it's done in collate function
        if self.encoder_input == 'context':
            encoder_input = context
        elif self.encoder_input == 'question':
            encoder_input = question
        elif self.encoder_input == 'context_prefix':
            context_splited = context.split()
            prefix_length = math.floor(len(context_splited) * self.prefix_ratio)
            encoder_input = " ".join(context_splited[:prefix_length])
        else: # self.encoder_input == 'retrieval_augmented_question':
            # raise NotImplementedError(f"Encoder input {self.encoder_input} is not implemented")
            # TODO: integrate retrieval
            retrieved_doc = self.retriever(question)
            encoder_input = question + self.tokenizer.sep_token + retrieved_doc
        encoder_inputs = self.tokenizer(encoder_input, truncation=True,
                                        max_length=self.max_seq_length,
                                        return_token_type_ids=True,
                                        return_special_tokens_mask=True,
                                        return_tensors='pt')
        # decoder label:
        # - context or context_label are for pretraining, where context returns original text, context_label returns
        #   mlm labels of the context
        # - answer is for downstream tasks
        if self.corrupt_text:
            encoder_input_tensors = (encoder_inputs['input_ids'], encoder_inputs['attention_mask'],
                                     encoder_inputs['token_type_ids'], encoder_inputs['special_tokens_mask'])
            mlm_inputs, mlm_labels = self.mlm_corrput_text(encoder_input_tensors)
            # TODO: attention_mask etc. of the encoder inputs should also be replaced
            encoder_inputs['input_ids'] = mlm_inputs

        if self.decoder_label == 'context_label':
            decoder_labels = mlm_labels
        else:
            if self.decoder_label == 'context':
                decoder_input = context
            elif self.decoder_label == 'answer':
                decoder_input = answer
            elif self.decoder_label == 'context_suffix':
                # As context_prefix and context_suffix are always used together, we should
                # be able to access the context_splited and prefix_length defined earlier
                decoder_input = " ".join(context_splited[prefix_length:])
            else:
                raise NotImplementedError(f"decoder_input {self.decoder_label} is not implemented")
            decoder_inputs = self.tokenizer(decoder_input, truncation=True,
                                            max_length=self.max_seq_length,
                                            return_tensors='pt')
            decoder_labels = decoder_inputs['input_ids']
        return encoder_inputs, decoder_labels

    def mlm_corrput_text(self, lm_tensors):
        """Adapted from process_lm_data in utils/data_utils.py L124
        
        Args:
            lm_tensors: a tuple of input_ids, input_mask, segment_ids, output_mask. Each is of shape [n_choice, max_seq_len]
        """
        input_ids, special_tokens_mask = lm_tensors[0], lm_tensors[3]
        assert input_ids.dim() == 2 and special_tokens_mask.dim() == 2
        _nc, _seqlen = input_ids.size()

        _inputs = input_ids.clone().view(-1, _seqlen) #remember to clone input_ids
        _mask_labels = []
        for ex in _inputs:
            if self.span_mask:
                _mask_label = self._span_mask(self.tokenizer.convert_ids_to_tokens(ex))
            else:
                _mask_label = self._word_mask(self.tokenizer.convert_ids_to_tokens(ex))
            _mask_labels.append(_mask_label)
        _mask_labels = torch.tensor(_mask_labels, device=_inputs.device)

        batch_lm_inputs, batch_lm_labels = self.mask_tokens(inputs=_inputs, mask_labels=_mask_labels, special_tokens_mask=special_tokens_mask.view(-1, _seqlen))

        batch_lm_inputs = batch_lm_inputs.view(_nc, _seqlen) #this is masked
        batch_lm_labels = batch_lm_labels.view(_nc, _seqlen)
        return batch_lm_inputs, batch_lm_labels

    def _span_mask(self, input_tokens: List[str], max_predictions=512):
        """Copid from _span_mask in utils/data_utils.py L251
        Get 0/1 labels for masking tokens at word level
        """
        effective_num_toks = 0
        cand_indexes = []
        if isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            after_special_tok = False
            for (i, token) in enumerate(input_tokens):
                if token in ["[CLS]", "[SEP]", "[PAD]"]:
                    after_special_tok = True
                    continue
                if len(cand_indexes) >= 1 and (not after_special_tok) and token.startswith("##"):
                    cand_indexes[-1].append(i)
                else:
                    cand_indexes.append([i])
                after_special_tok = False
                effective_num_toks += 1
        elif isinstance(self.tokenizer, (RobertaTokenizer, RobertaTokenizerFast)):
            after_special_tok = False
            for (i, token) in enumerate(input_tokens):
                if token in ["<s>",  "</s>", "<pad>"]:
                    after_special_tok = True
                    continue
                if len(cand_indexes) >= 1 and (not after_special_tok) and (not token.startswith("Ġ")):
                    cand_indexes[-1].append(i)
                else:
                    cand_indexes.append([i])
                after_special_tok = False
                effective_num_toks += 1
        else:
            raise NotImplementedError
        cand_indexes_args = list(range(len(cand_indexes)))

        random.shuffle(cand_indexes_args)
        num_to_predict = min(max_predictions, max(1, int(round(effective_num_toks * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for wid in cand_indexes_args:
            if len(masked_lms) >= num_to_predict:
                break
            span_len = np.random.choice(self.span_lens, p=self.span_len_dist)
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            # if len(masked_lms) + span_len > num_to_predict:
            #     continue
            index_set = []
            is_any_index_covered = False
            for _wid in range(wid, len(cand_indexes)): #iterate over word
                if len(index_set) + len(cand_indexes[_wid]) > span_len:
                    break
                for _index in cand_indexes[_wid]: #iterate over subword
                    if _index in covered_indexes:
                        is_any_index_covered = True
                        break
                    index_set.append(_index)
                if is_any_index_covered:
                    break
            if is_any_index_covered:
                continue
            for _index in index_set:
                covered_indexes.add(_index)
                masked_lms.append(_index)

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def _word_mask(self, input_tokens: List[str], max_predictions=512):
        """Copid from _word_mask in utils/data_utils.py L192
        Get 0/1 labels for masking tokens at word level
        """
        effective_num_toks = 0
        cand_indexes = []
        after_special_tok = False
        for (i, token) in enumerate(input_tokens):
            if token in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                after_special_tok = True
                continue
            if isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
                # In Bert tokenizer, the following subword tokens of a word starts with "##"
                if len(cand_indexes) >= 1 and (not after_special_tok) and token.startswith("##"):
                    cand_indexes[-1].append(i)
                else:
                    cand_indexes.append([i])
            elif isinstance(self.tokenizer, (RobertaTokenizer, RobertaTokenizerFast)):
                # In Roberta tokenizer, the first subword token of a word starts with "Ġ"
                if len(cand_indexes) >= 1 and (not after_special_tok) and (not token.startswith("Ġ")):
                    cand_indexes[-1].append(i)
                else:
                    cand_indexes.append([i])
            elif isinstance(self.tokenizer, (T5Tokenizer, T5TokenizerFast)):
                # In T5 tokenizer, the first subword token of a word starts with "▁"
                if len(cand_indexes) >= 1 and (not after_special_tok) and not token.startswith("▁"):
                    cand_indexes[-1].append(i)
                else:
                    cand_indexes.append([i])
            else:
                raise NotImplementedError()
            after_special_tok = False
            effective_num_toks += 1
        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(effective_num_toks * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def mask_tokens(self, inputs, mask_labels, special_tokens_mask=None):
        """Copied from mask_tokens in utils/data_utils.py L149
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        
        Returns:
            masked_inputs: torch.Tensor of shape (batch_size, seq_len)
            labels: torch.Tensor of shape (batch_size, seq_len). It is initialized as orginal inputs,
                and then the masked input tokens are replaced by -100.
        """
        assert inputs.size() == mask_labels.size()

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        # if self.tokenizer._pad_token is not None: #should be handled already
        #     padding_mask = labels.eq(self.tokenizer.pad_token_id)
        #     probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=labels.device)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=labels.device)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, device=labels.device)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    
class Retriever(ABC):
    """Retriever base class.
    """
    @abstractmethod
    def __init__(self, n_docs=1):
        self.n_docs = n_docs

    @abstractmethod
    @lru_cache(maxsize=None)
    def retrieve_contexts(self, question):
        """Retrieve contexts for a given question.

        Args:
            question (str): The question to retrieve contexts for.
            aggregate_docs_fn (callable(list -> str), optional): Function to aggregate the retrieved documents.

        Retruns:
            contexts (list): The retrieved contexts.
        """
        pass

    def __call__(self, question):
       return self.retrieve_contexts(question)


def create_dummy_graph(batch_size):
    node_ids = torch.ones((batch_size, 1, 1), dtype=torch.long)
    node_type_ids = torch.full_like(node_ids, 3)
    node_scores = torch.rand((batch_size, 1, 1, 1), dtype=torch.float32)
    adj_lengths = torch.ones((batch_size, 1), dtype=torch.long)
    special_nodes_mask = torch.full((batch_size, 1, 1), False)
    edge_index = [[torch.zeros((2, 0), dtype=torch.long)] for _ in range(batch_size)]
    edge_type = [[torch.zeros((0,), dtype=torch.long)] for _ in range(batch_size)]
    pos_triples = [[[torch.zeros((0), dtype=torch.long) for _ in range(3)]] for _ in range(batch_size)]
    neg_nodes = [[torch.zeros((0, 0), dtype=torch.long)] for _ in range(batch_size)]
    # TODO: do this in one step
    node_ids = node_ids.squeeze(1)
    node_type_ids = node_type_ids.squeeze(1)
    node_scores = node_scores.squeeze(1)
    adj_lengths = adj_lengths.squeeze(1)
    special_nodes_mask = special_nodes_mask.squeeze(1)
    edge_index = sum(edge_index,[])
    edge_type = sum(edge_type,[])
    pos_triples = sum(pos_triples,[])
    neg_nodes = sum(neg_nodes,[])
    return node_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask, \
        edge_index, edge_type, pos_triples, neg_nodes


def se2seq_collate_texts(input_ids, attention_mask, token_type_ids, special_tokens_mask,
                  decoder_labels, *, tokenizer):
    """
    The inputs are assumed to have the shape of [batch_size, 1, seq_len]
    The outputs removes the choice dimension, and pad the sequences to the same length.
    Returns:
        a dictionary with keys:
            - input_ids: [batch_size, seq_len]
            - attention_mask: [batch_size, seq_len]
            - token_type_ids: [batch_size, seq_len]
            - special_tokens_mask: [batch_size, seq_len]
            - labels: [batch_size, seq_len]
    """
    collator = DataCollatorForSeq2Seq(tokenizer, max_length=256)
    # Squeeze the choice dimension
    inputs = [input_ids, attention_mask, token_type_ids, special_tokens_mask,
                  decoder_labels]
    for i, item in enumerate(inputs):
        if item is not None:
            inputs[i] = [e[0] for e in item]
    input_ids, attention_mask, token_type_ids, special_tokens_mask,\
        decoder_labels = inputs
    features = [
        {
            'input_ids': ii,
            'attention_mask': am,
            'token_type_ids': tyi,
            'special_tokens_mask': stm,
            'labels': dl,
        }
        for ii, am, tyi, stm, dl in zip(input_ids, attention_mask, token_type_ids,
                                        special_tokens_mask, decoder_labels)
    ]
    features = collator(features)
    return features['input_ids'], features['attention_mask'], \
        features['token_type_ids'], features['special_tokens_mask'],\
        features['labels']


def dragon_collate_fn(examples, tokenizer,dummy_graph=False):
    """For DragonDataset of encoder-decoder architecture.
    """
    # Tensors of shape [batch_size, num_choices, max_seq_length] (written separately for clarity)
    input_ids = [example[0] for example in examples]
    attention_mask = [example[1] for example in examples]
    token_type_ids = [example[2] for example in examples]
    special_token_mask = [example[3] for example in examples]
    decoder_labels = [example[4] for example in examples]
    # The texts were tokenized with return_tensors='pt', so they have an extra first dimension 
    (input_ids, attention_mask, token_type_ids, special_token_mask,
     decoder_labels) = se2seq_collate_texts(input_ids, attention_mask,
                                            token_type_ids, special_token_mask,
                                            decoder_labels=decoder_labels,
                                            tokenizer=tokenizer)
    if dummy_graph:
        batch_size = input_ids.size(0)
        node_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask, \
        edge_index, edge_type, pos_triples, neg_nodes = create_dummy_graph(batch_size)
    else:
        # Tensors of shape [batch_size, num_choices, max_nodes]
        node_ids = torch.cat([example[5] for example in examples], dim=0)
        node_type_ids = torch.cat([example[6] for example in examples], dim=0)
        # A tensor of shape [batch_size, num_choices, max_nodes, 1]
        node_scores = torch.cat([example[7] for example in examples], dim=0)
        # A tensor of shape [batch_size, 1]
        adj_lengths = torch.cat([example[8] for example in examples], dim=0)
        # A tensor of shape [batch_size, num_choices, max_nodes]
        special_nodes_mask = torch.cat([example[9] for example in examples], dim=0)
        # Aggregated by wrapping into a list
        edge_index = [example[10][0] for example in examples]
        edge_type = [example[11][0] for example in examples]
        pos_triples = [example[12][0] for example in examples]
        neg_nodes = [example[13][0] for example in examples]
    return input_ids, attention_mask, token_type_ids, special_token_mask, decoder_labels, \
        node_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask, \
        edge_index, edge_type, pos_triples, neg_nodes


def load_data(args, dataset_cls, collate_fn, corrupt=True, num_workers=1, dummy_graph=False,
              dataset_kwargs=dict()):
    """Construct the dataset and return dataloaders

    Args:
        dataset_cls (class): DragonDataset or DragonEncDecDataset
        collate_fn (callable): dragond_collate_fn, dragond_adapt2enc_collate_fn or dragond_encdec_collate_fn
        corrupt (bool): whether to corrupt the graph and text
        num_workers (int): number of workers for dataloader
    Returns:
        train_dataloader, dev_dataloader, test_dataloader
    """
    if dummy_graph:
        collate_fn = partial(collate_fn, dummy_graph=True)
    num_relations = args.num_relations
    model_name = args.encoder_name_or_path
    max_seq_length = args.max_seq_len
    train_dataset = dataset_cls(statement_path=args.train_statements, adj_path=args.train_adj, legacy_adj=args.legacy_adj,
                                num_relations=num_relations, corrupt_graph=corrupt, corrupt_text=corrupt, model_name=model_name,
                                max_seq_length=max_seq_length, **dataset_kwargs)
    dev_dataset = dataset_cls(statement_path=args.dev_statements, adj_path=args.dev_adj, legacy_adj=args.legacy_adj,
                              num_relations=num_relations, corrupt_graph=corrupt, corrupt_text=corrupt, model_name=model_name,
                              max_seq_length=max_seq_length, **dataset_kwargs)
    test_dataset = dataset_cls(statement_path=args.test_statements, adj_path=args.test_adj, legacy_adj=args.legacy_adj,
                               num_relations=num_relations, corrupt_graph=corrupt, corrupt_text=corrupt, model_name=model_name,
                               max_seq_length=max_seq_length, **dataset_kwargs)
    # set tokenizer
    collate_fn = partial(collate_fn, tokenizer=train_dataset.tokenizer)
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size,num_workers=num_workers)
    dev_dataloader = DataLoader(dev_dataset, collate_fn=collate_fn, batch_size=args.eval_batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=args.eval_batch_size, num_workers=num_workers)
    return train_dataloader, dev_dataloader, test_dataloader
