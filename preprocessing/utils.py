"""
This script contains functions for easily creating dataset. It provides a
unified interface for data preprocessing.
"""
import json
import os
from pathlib import Path

import srsly
from tqdm import tqdm

from preprocess_utils.wikidata.grounding import ground
from preprocess_utils.wikidata.subgraph import \
    generate_adj_data_from_grounded_entities

# Constants for creating subgraph
WIKIDATA_GRAPH_PATH = 'data/wikidata5m/wikdata5m.graph'
WIKIDATA_UNDIRECTED_GRAPH_PATH = 'data/wikidata5m/wikdata5m_undirected.graph'
WIKIDATA_TRANSE_EMBEDDING_PATH = 'data/wikidata5m/graphvite-transe-wikidata5m.pkl'
GLOVE_EMBEDDING_PATH = 'data/glove/glove.6B/glove.6B.50d.txt'
ENTITY2ALIASES_PATH = 'data/wikidata5m/entity2aliases.pkl'
RELATION2ALIASES_PATH = 'data/wikidata5m/relation2aliases.pkl'





def create_subgraph(ground_paths, split_names, dataset_root, file_prefix="", num_process=1,
                    scoring='length', extend_one_hop=True):
    graph_paths = {}
    for split in split_names:
        ground_path = ground_paths[split]
        output_path = os.path.join(dataset_root, f'adj_data/{file_prefix}{split}.adj_data.pkl')
        generate_adj_data_from_grounded_entities(ground_path,
            graph_path=WIKIDATA_GRAPH_PATH,
            undirected_graph_path=WIKIDATA_UNDIRECTED_GRAPH_PATH,
            embedding_path=WIKIDATA_TRANSE_EMBEDDING_PATH,
            output_path=output_path,
            num_process=num_process,
            scoring=scoring,
            glove_embedding_path=GLOVE_EMBEDDING_PATH,
            entity2aliases_path=ENTITY2ALIASES_PATH,
            relation2aliases_path=RELATION2ALIASES_PATH,
            prune_oos_nodes=True,
            extend=extend_one_hop)
        graph_paths[split] = output_path
    return graph_paths


def read_statements(statement_path):
    with open(statement_path, "r", encoding="utf-8") as f:
        statements = []
        for line in tqdm(f.readlines()):
            json_dic = json.loads(line)
            question = json_dic["question"]["stem"]
            # The statements are still in dragon's format, therefore there is a choices dimension.
            answer = json_dic["question"]["choices"][0]["text"]
            # TODO: There will be more than one contexts in the future.
            contexts = json_dic["context"]
            statements.append({
                    "id": json_dic["id"],
                    "contexts": contexts,
                    "question": question,
                    "endings": answer,})
    return statements