"""This script profiles the number of nodes and edges of the train dataset."""
import os
import sys
# Add parent directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import pandas as pd
from tqdm import tqdm

from utils.common import load_args
from dataset.lmgnn import load_data


def get_sequence_and_node_dataframes(dataset):
    seq_lens = []
    num_nodes = []
    num_edges = []
    for sample in tqdm(dataset, desc="Profiling train dataset"):
        input_ids, attention_mask, decoder_labels,\
            node_ids, node_type_ids, adj_length,\
            edge_index, edge_type = sample
        seq_lens.append(input_ids.shape[1])
        num_nodes.append((node_ids != 1).sum().item())
        num_edges.append(edge_index[0].shape[1])
    seq_lens = pd.Series(seq_lens)
    num_nodes = pd.Series(num_nodes)
    num_edges = pd.Series(num_edges)
    return seq_lens, num_nodes, num_edges


def main(args):
    args = load_args(config_path=args.config_path, profile=args.profile)
    train_loader, _ = load_data(
        args,
        train_kwargs={'encoder_input': args.encoder_input, 'decoder_label': args.decoder_label})
    train_dataset = train_loader.dataset
    seq_lens, num_nodes, num_edges = get_sequence_and_node_dataframes(train_dataset)
    for info, df in zip(["Sequence lengths", "Number of nodes", "Number of edges"],
                        [seq_lens, num_nodes, num_edges]):
        print("-" * 60)
        print(info.upper() + ":")
        print(df.describe())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='configs/lmgnn.yaml')
    parser.add_argument('--profile', type=str, default='finetune_mcwq')
    args = parser.parse_args()
    main(args)
