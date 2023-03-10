import pickle

import pytest
import torch
from transformers.optimization import Adafactor

from dataset.lmgnn import load_data
from models.flamingo_t5 import FlamingoConfig, FlamingoT5Decoder
from models.t5_lmgnn import T5GNNConfig, T5GNNEncoder
from models.t5_seq2seq import T5Seq2Seq
from utils.common import load_args
from utils.model_utils import get_tweaked_num_relations

DUMMY_BATCH_PATH = 'tests/dummy_batch.pkl'

@pytest.fixture
def args():
    args = load_args()
    args.batch_size = 2
    args.encoder_name_or_path = 't5-small'
    args.k = 2  # gnn layerss
    return args

@pytest.fixture
def encoder(args):
    # a dummy node embedding of 1000 nodes, with dim=512 each 
    node_emb = torch.randn((1000, 512))
    config = T5GNNConfig(
        encoder_name_or_path=args.encoder_name_or_path,
        gnn_dim=args.gnn_dim,
        ie_dim=args.ie_dim,
        node_in_dim=512,
        n_ntype=4,
        n_etype=get_tweaked_num_relations(args.num_relations, args.cxt_node_connects_all) * 2,
        num_ie_layer=args.ie_layer_num,
        num_gnn_layers=args.k,
        num_entity=1000,
        dropout_gnn=0.1,
        dropout_emb=0.1,
    )
    encoder = T5GNNEncoder(config, pretrained_node_emb=node_emb)
    for n, p in encoder.named_parameters():
        if 'node_emb.emb' in n:
            p.retain_grad = False
    return encoder


@pytest.fixture
def decoder(args, encoder):
    decoder_config = FlamingoConfig(
        d_model=encoder.config.d_model,
        dim_media=100,
        xattn_dim_head=8,
        xattn_heads=1,
        xattn_every=1,
        xattn_ff_mult=4,
        lm_name_or_path=args.encoder_name_or_path,)
    decoder = FlamingoT5Decoder(decoder_config, encoder.get_input_embeddings())
    return decoder


@pytest.fixture
def model(encoder, decoder):
    model = T5Seq2Seq(
        encoder=encoder,
        decoder=decoder)
    return model


@pytest.fixture
def batch(args):
    train_loader, _ = load_data(args, corrupt=False)
    batch = next(iter(train_loader))
    (input_ids, attention_mask, decoder_labels,
        node_ids, node_type_ids, adj_lengths,
        edge_index, edge_type) = batch
    node_ids = torch.ones_like(node_ids)
    batch = (input_ids, attention_mask, decoder_labels,
        node_ids, node_type_ids, adj_lengths,
        edge_index, edge_type)
    return batch


@pytest.fixture
def dummy_batch():
    with open(DUMMY_BATCH_PATH, "rb") as f:
        dummy_batch = pickle.load(f)
        return dummy_batch


def test_forward(dummy_batch, model):
    (input_ids, attention_mask, decoder_labels,
        node_ids, node_type_ids, adj_lengths,
        edge_index, edge_type) = dummy_batch
    with torch.no_grad():
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=decoder_labels,
            node_ids=node_ids,
            node_type_ids=node_type_ids,
            adj_lengths=adj_lengths,
            edge_index=edge_index,
            edge_type=edge_type,
            return_dict=True
        )


def test_backward(dummy_batch, model):
    (input_ids, attention_mask, decoder_labels,
        node_ids, node_type_ids, adj_lengths,
        edge_index, edge_type) = dummy_batch
    optimizer = Adafactor(model.parameters(), lr=0.001, relative_step=False)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=decoder_labels,
        node_ids=node_ids,
        node_type_ids=node_type_ids,
        adj_lengths=adj_lengths,
        edge_index=edge_index,
        edge_type=edge_type,
        return_dict=True
    )
    loss = outputs.loss
    loss.backward()
    optimizer.step()
