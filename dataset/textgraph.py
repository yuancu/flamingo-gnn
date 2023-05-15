"""
In this module, we embed the texts of the nodes, and use them as node embeddings.
"""
import math

import numpy as np
import srsly
import torch
from scipy.sparse import coo_matrix
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer

from utils.model_utils import get_tweaked_num_relations


class LabelEncoder(torch.nn.Module):
    """A lightning module that wraps a sentence encoder."""
    def __init__(self, model_name_or_path, pool='cls'):
        if pool not in ['cls', 'avg']:
            raise ValueError(f"pool method must be either cls or avg, got {pool}")
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path)
        if self.model.config.is_encoder_decoder:
            self.model = self.model.encoder
            print("The model is an encoder-decoder model, only the encoder will be used.")
        self.config = self.model.config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.pool_method = pool

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @staticmethod
    def cls_pool(last_hidden_states, attention_mask=None):
        """CLS pool the sentence embedding.
        This is the pooling method adopted by RUC's SR paper.

        Args:
            last_hidden_states: [..., seq_len, embedding_dim]
            attention_mask: [..., seq_len] silently ignored!
                It exists for compatibility with other pooling methods.

        Returns:
            torch.Tensor: pooled_embedding [..., embedding_dim]
        """
        return last_hidden_states[..., 0, :]

    @staticmethod
    def avg_pool(last_hidden_states, attention_mask=None):
        """Average pool the sentence embedding.

        Args:
            last_hidden_states (torch.Tensor): [..., seq_len, embedding_dim]
            attention_mask (torch.Tensor): [..., seq_len]

        Returns:
            torch.Tensor: pooled_embedding [..., embedding_dim]
        """
        # Compute the average embedding, ignoring the padding tokens.
        if attention_mask is None:
            attention_mask = torch.ones(last_hidden_states.shape[:-1], device=last_hidden_states.device)
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=-2) / attention_mask.sum(dim=-1)[..., None]

    def encode(self, text):
        """Pool the query and target(s) sentence embeddings.

        Args:
            text (str | list[str]): a list of texts

        Returns:
            torch.Tensor: pooled sentence embeddings
        """
        tokenized = self.tokenizer(text, return_tensors='pt')
        embeddings = self.model(**tokenized).last_hidden_state
        if self.pool_method == 'cls':
            embeddings = self.cls_pool(embeddings) # [..., 1 + k, embedding_dim]
        else:
            attention_mask = tokenized['attention_mask']  # [..., 2 + k, seq_len]
            embeddings = self.avg_pool(embeddings, attention_mask)
        return embeddings


class GraphInputEncoder(torch.nn.Module):
    """Encode the labels of the input to a single embedding vector."""
    def __init__(self, model_name_or_path, pool='cls') :
        super().__init__()
        self.label_encoder = LabelEncoder(model_name_or_path, pool=pool)

    def encode(self, entity_labels, relation_labels):
        n_entity = len(entity_labels)
        embeddings = self.label_encoder.encode(entity_labels + relation_labels)
        entity_embeddings = embeddings[:n_entity]
        relation_embeddings = embeddings[n_entity:]
        return entity_embeddings, relation_embeddings

    def forward(self, entity_labels, relation_labels):
        return self.encode(entity_labels, relation_labels)


class TextGraphDataset(Dataset):
    """Convert the dataset to graph embeddings.
    
    The input file should be a jsonl file, with each line containing the following fields:
    - id
    - question
    - context (optional)
    - answers
    - entities
    - adjacency
    - entity_labels
    
    An example:
    ```json
    {
        "id": "130578",
        "question": "What American Mexican cinematographer did Brokeback Mountain star",
        "context": "",
        "answers": ["Rodrigo Prieto"],
        "entities": ["Q30", "Q1341403", "Q222344", "Q96", "Q160618"],
        "relations": ["P106", "P161", "P27"],
        "adjacency": [
            [1, 0, 2],
            [4, 1, 1],
            [1, 2, 0],
            [1, 2, 3]
        ],
        "entity_labels": ["unite states of america", "rodrigo prieto", "cinematographers",
            "mexican federal republic", "brokeback mountain (film)"]
    }
    ```
    """
    def __init__(self, dataset_path, relation2id, model_name='t5-base', max_seq_length=256,
                 encoder_input='qustion', decoder_label='answer', prefix_ratio=0.2,
                 num_relations=820, cxt_node_connects_all=True, max_node_num=100):
        """Create a TextGraphDataset.

        Args:
            dataset_path (str): The path to the input jsonl file.
            relation2id (dict): a dict mapping relation text identifiers to indices.
        """
        super().__init__()
        self.examples = list(srsly.read_jsonl(dataset_path))
        self.relation2id = relation2id
        self.encoder_input = encoder_input
        self.decoder_label = decoder_label
        self.prefix_ratio = prefix_ratio
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_length = max_seq_length
        # Graph attributes
        self.num_relations = num_relations
        self.cxt_node_connects_all = cxt_node_connects_all
        self.max_node_num = max_node_num

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        # 1. Load text data
        # keys: id, question, answers, entities, adjacency, entity_labels
        example = self.examples[index]
        # MLM is also done in postprocess_text
        encoder_inputs, decoder_labels = self.process_text(example)
        input_ids, attention_mask = encoder_inputs["input_ids"], encoder_inputs["attention_mask"]

        # 2. Load graph data, post processing is done in the _load_graph_from_index function
        node_ids, node_type_ids, adj_length,\
            edge_index, edge_type = self.process_graph(index)

        return input_ids, attention_mask, decoder_labels,\
            node_ids, node_type_ids, adj_length,\
            edge_index, edge_type

    def process_text(self, example):
        """Adapted from load_input_tensors in utils/data_utils.py L584
        Args:
            example: an InputExample object
        Returns:
            encoder_inputs (dict): a dictionary containing input_ids and attention_mask
            decoder_labels (tensor | list): input_ids for decoder, or a list of raw strings as answers
        """
        if 'context' in example:
            assert isinstance(example['context'], str)
        if 'question' in example:
            assert isinstance(example['question'], str)
        if 'answers' in example:
            assert isinstance(example['answers'], list)
        # Here we separately encode the question and answer
        # Warning: we assume the encoder and the decoder share the same tokenizer
        # but this is not necessarily true!
        # Padding is removed here, it's done in collate function
        if self.encoder_input == 'context':
            encoder_input = example['context']
        elif self.encoder_input == 'question':
            encoder_input = 'question: ' + example['question']
        elif self.encoder_input == 'contextualized_question':
            encoder_input = 'question: ' + example['question'] + ' context: ' + example['context']
        elif self.encoder_input == 'context_prefix':
            context_splited = example['context'].split()
            prefix_length = math.floor(len(context_splited) * self.prefix_ratio)
            encoder_input = "complete: " +  " ".join(context_splited[:prefix_length])
        else: # self.encoder_input == 'retrieval_augmented_question':
            raise NotImplementedError(f"Encoder input {self.encoder_input} is not implemented")
        encoder_inputs = self.tokenizer(encoder_input, truncation=True,
                                        max_length=self.max_seq_length,
                                        return_tensors='pt')
        # decoder label:
        # - context or context_label are for pretraining, where context returns original text, context_label returns
        #   mlm labels of the context
        # - answer is for downstream tasks
        if self.decoder_label == self.encoder_input:
            decoder_labels = encoder_inputs['input_ids']
        elif self.decoder_label == 'raw_answers':
            decoder_labels = example['answers']
        else:
            if self.decoder_label == 'context':
                decoder_input = example['context']
            elif self.decoder_label == 'answer':
                decoder_input = example['answers'][0]
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

    def process_graph(self, example):
        """Create node and edge features from the example
        
        Used keys in example:
        - entities
        - relations
        - adjacency
        - entity_labels

        Args:
            example (dict): input example

        Returns:
            tuple: node_ids, node_type_ids, adj_length, edge_index, edge_type
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

        adj = self.triplet2adj(example['entities'], example['relations'], example['adjacency'])
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
        
        if len(extra_i) > 0:
            i = torch.cat([i, torch.tensor(extra_i)], dim=0)
            j = torch.cat([j, torch.tensor(extra_j)], dim=0)
            k = torch.cat([k, torch.tensor(extra_k)], dim=0)
        mask = (j < actual_max_node_num) & (k < actual_max_node_num)
        i, j, k = i[mask], j[mask], k[mask]
        i, j, k = torch.cat((i, i + half_n_rel), 0), torch.cat((j, k), 0), torch.cat((k, j), 0)  # add inverse relations
        edge_index = torch.stack([j,k], dim=0) # each entry is [2, E]
        edge_type = i # each entry is [E, ]
        edge_index = [edge_index]
        edge_type = [edge_type]
        node_ids, node_type_ids, adj_length = [x.view(1, *x.size()) for x in ( node_ids, node_type_ids, adj_length)]
        return node_ids, node_type_ids, adj_length, (edge_index, edge_type)

    def triplet2adj(self, entities, relations, triplets):
        """Convert triplets of [head_local_idx, relation_local_idx, tail_local_idx]
        to a sparse adjacency matrix, with entity indicies being local and relation
        indices being global.

        Args:
            entities: list[str]
                entitiy identifiers
            relations: list[str]
                relation identifiers
            triplets: list[list[int, int, int]]
                each element is a triple of (head, relation, tail), where the
                head and tail are the indices in the entities, and relation is
                the relation index in relations.

        Returns:
            adjacency: sparse matrix (n_rel * n_node, n_node)
                adjacency matrix of the graph
        """
        # In the current algorithm, entities are local, while relations are global.
        # this means the location of entities are related to their local index,
        n_node = len(entities)
        # while the location of relations are related to their global index.
        n_rel = len(self.relation2id)
        adjacency_matrix = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)

        for edge in triplets:
            head, relation, tail = edge
            if relations[relation] in self.relation2id:
                adjacency_matrix[self.relation2id[relations[relation]]][head][tail] = 1
        if n_node == 0:
            adjacency_matrix = coo_matrix((n_rel * n_node, n_node), dtype=np.uint8)
        else:
            adjacency_matrix = coo_matrix(adjacency_matrix.reshape(-1, n_node))
        return adjacency_matrix
