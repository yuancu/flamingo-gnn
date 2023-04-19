"""
This script Extract and Transform (but not really Load) the subgraphs retrieved by the srtk module to the format
that out module requires.

SRTK format: A jsonl file where each line is a json object with the following fields of interest:
- id
- question
- triplets


The format that out module requires: Each sample is a pickle file of a tuple
- adjacency: scipy.sparse.coo_matrix
- nodes: 
- question mask
- answer mask

Additionally, a statement jsonl file corresponding to the subgraphs is required.
Each line of it is a json object with the following fields of interest:
- id
- question
- answers
This file should be prepared separately.
"""
import argparse
import numpy as np
import pickle
from pathlib import Path

import srsly
from scipy.sparse import coo_matrix
from tqdm import tqdm


def create_adjacency_matrix(triplets, relation2id, entity2id):
    """
    Args:
        edges: list[tuple]
            list of (head, relation, tail) triplets, where the head is always wikidata entity, while the tail is
            either a wikidata entity, or an attribute like date.
        relation2id: dict[str: int]
            a dictionary of relation to its id mapping
    Returns:
        adjacency: sparse matrix (n_rel * n_node, n_node)
            adjacency matrix of the graph
        qids: list[int]
            basically a mapping from the node index in the subgraph to the qid.
    """
    head_qids = set([triplet[0] for triplet in triplets])
    tail_qids = set([triplet[-1] for triplet in triplets])
    for qid in head_qids.union(tail_qids):
        assert qid.startswith('Q')
    qids = list(head_qids.union(tail_qids))
    n_node = len(qids)
    n_rel = len(relation2id)
    adjacency = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)

    # speedup index lookup, node_map is {str: i}
    node_map = {node: i for i, node in enumerate(qids)}
    for triplet in triplets:
        head, relation, tail = triplet
        adjacency[relation2id[relation]][node_map[head]][node_map[tail]] = 1
    if n_node == 0:
        adjacency = coo_matrix((n_rel * n_node, n_node), dtype=np.uint8)
    else:
        adjacency = coo_matrix(adjacency.reshape(-1, n_node))
    # convert nodes to their integer representation. 
    related_nodes = [entity2id[qid] for qid in qids]
    return adjacency, related_nodes


def create_masks(qids): 
    """Create question mask and answer mask for the subgraph. In this implementation,
    It returns a mask of all True for question mask and a mask of all False for answer mask.

    This exists mainly to conform to the format of the dragon GNN module.
    The DRAGON was used for node classification, therefore the answer entities
    are filtered out with a mask. In our case, we utilize the semantic information
    of the retrieve subgraphs, therefore we don't need to filter out the answer entities.
    """
    question_mask = np.full(len(qids), True)
    answer_mask = np.full(len(qids), False)
    return question_mask, answer_mask


def create_subgraph_entry(triplets, relation2id, entity2id):
    """Create a subgraph tuple consisting of the adjacency matrix, related nodes in indices and question
    and answer masks."""
    adj, related_nodes = create_adjacency_matrix(triplets, relation2id, entity2id)
    qmask, amask = create_masks(related_nodes)
    return adj, related_nodes, qmask, amask


def main(args):
    with open(args.relation2id, 'rb') as f:
        relation2id = pickle.load(f)
    with open(args.entity2id, 'rb') as f:
        entity2id = pickle.load(f)
    adjacency_dir = Path(args.outut_dir) / 'adj'
    if not adjacency_dir.exists():
        adjacency_dir.mkdir(parents=True)
        print(f'Created adjacency directory {adjacency_dir}')
    subgraphs = srsly.read_jsonl(args.input)
    total = sum(1 for _ in srsly.read_jsonl(args.input))
    for subgraph in tqdm(subgraphs, total=total, desc="Creating subgraph pickles"):
        sample_id = subgraph['id']
        adj, related_nodes, qmask, amask = create_subgraph_entry(subgraph['triplets'], relation2id, entity2id)
        # Save the adjacency matrix
        with open(adjacency_dir / f'{sample_id}.pkl', 'wb') as f:
            pickle.dump((adj, related_nodes, qmask, amask), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='path to the retrieved subgraph jsonl file, each\
                        line is expected to be a json object with id, question and triplets')
    parser.add_argument('-o', '--outut-dir', required=True, help='path to the output directory')
    parser.add_argument('--relation2id', type=str, required=True, help='path to the relation2id pickle file')
    parser.add_argument('--entity2id', type=str, required=True, help='path to the entity2id pickle file')
    args = parser.parse_args()
    
    main(args)
