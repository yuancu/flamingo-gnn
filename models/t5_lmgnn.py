"""
This is built on top of LMGNN and DRAGON model.
This module patches GNN into the encoder of T5 model.
"""
import copy
import inspect
import logging
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig
from transformers.modeling_outputs import (BaseModelOutput,
                                           ModelOutput, )
from transformers.models.t5.modeling_t5 import (T5EncoderModel, T5Stack)

from .gnn import GATConvE, TransEDecoder, DistMultDecoder, RotatEDecoder, make_one_hot
from utils.model_utils import construct_encoder
from utils.layers import MLP, CustomizedEmbedding, MultiheadAttPoolLayer

logger = logging.getLogger(__name__)



@dataclass
class DragonEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    pooled_language_representation: torch.FloatTensor = None
    pooled_gnn_representation: torch.FloatTensor = None
    link_losses: Union[List, tuple] = None
    gnn_hidden_states: Optional[torch.FloatTensor] = None


@dataclass
class T5GATOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    node_features: Optional[torch.FloatTensor] = None


class T5DragonEncoder(nn.Module):
    def __init__(self, args=Namespace(), model_name="t5-base", k=5, n_ntype=4, n_etype=38, n_node=799273, node_dim=200,
                 node_in_dim=1024, n_attention_head=2, fc_dim=200, n_fc_layer=0, p_emb=0.2, p_gnn=0.2, p_fc=0.2,
                 pretrained_node_emb=None, freeze_ent_emb=True, init_range=0.02, ie_dim=200, info_exchange=True,
                 ie_layer_num=1, sep_ie_layers=False, layer_id=-1):
        super().__init__()
        self.n_ntype = n_ntype
        self.n_etype = n_etype
        self.lmgnn, self.loading_info = T5GNN.from_pretrained(model_name,
            output_loading_info=True, args=args, k=k, n_ntype=n_ntype, n_etype=n_etype,
            n_node=n_node, node_dim=node_dim, node_in_dim=node_in_dim,
            n_attention_head=n_attention_head, fc_dim=fc_dim, n_fc_layer=n_fc_layer,
            p_emb=p_emb, p_gnn=p_gnn, p_fc=p_fc, pretrained_node_emb=pretrained_node_emb,
            freeze_ent_emb=freeze_ent_emb, init_range=init_range, ie_dim=ie_dim,
            info_exchange=info_exchange, ie_layer_num=ie_layer_num,  sep_ie_layers=sep_ie_layers,
            layer_id=layer_id)
        backbone_config = self.lmgnn.backbone.config
        self.vocab_size = backbone_config.vocab_size
        # Caution: this might be wrong!
        self.config = self.lmgnn.config
        # This setting is for generation
        self.main_input_name = self.lmgnn.backbone.main_input_name

    def batch_graph(self, edge_index_init, edge_type_init, pos_triples_init, neg_nodes_init, n_nodes):
        """
        edge_index_init:  list of (n_examples, ). each entry is torch.tensor(2, E?)    ==> [2, total_E]
        edge_type_init:   list of (n_examples, ). each entry is torch.tensor(E?, )     ==> [total_E, ]
        pos_triples_init: list of (n_examples, ). each entry is [h,r,t] where h/r/t: torch.tensor(n_triple?, ) ==> [3, `total_n_triple`]
        neg_nodes_init:   list of (n_examples, ). each entry is torch.tensor(n_triple?, n_neg) ==> [`total_n_triple`, n_neg]
        """
        n_examples = len(edge_index_init)
        edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        edge_index = torch.cat(edge_index, dim=1) #[2, total_E]
        edge_type = torch.cat(edge_type_init, dim=0) #[total_E, ]

        pos_triples = [[], [], []]
        for _i_ in range(n_examples):
            h = pos_triples_init[_i_][0] + _i_ * n_nodes #tensor[n_triple?,]
            r = pos_triples_init[_i_][1]                 #tensor[n_triple?,]
            t = pos_triples_init[_i_][2] + _i_ * n_nodes #tensor[n_triple?,]
            pos_triples[0].append(h)
            pos_triples[1].append(r)
            pos_triples[2].append(t)
        pos_triples = torch.stack([torch.cat(item) for item in pos_triples]) #[3, `total_n_triple`] where `total_n_triple` is sum of n_triple within batch
        assert pos_triples.size(0) == 3

        neg_nodes = [neg_nodes_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        neg_nodes = torch.cat(neg_nodes) #[`total_n_triple`, n_neg]
        assert neg_nodes.dim() == 2
        assert pos_triples.size(1) == neg_nodes.size(0)
        return edge_index, edge_type, pos_triples, neg_nodes

    def forward(self, input_ids, attention_mask, token_type_ids, output_mask, node_ids,
                node_type_ids, node_scores, adj_lengths, special_nodes_mask, edge_index, edge_type,
                pos_triples, neg_nodes, inputs_embeds=None, output_hidden_states=True, output_attentions=True,
                cache_output=False, return_dict=True):
        """This method should fit the encoder in EncoderDecoderModel.
        Should take in at least: input_ids, attention_mask, inputs_embeds, output_attentions,
        outputput_hidden_states, return_dict.
        The output should fit BaseModelOutput: it should return last_hidden_state, hidden_states, attentions
        NOTE: the input format, shape, and outputs are different from the original DRAGON model.

        Args:
            input_ids: (batch_size, seq_len) in pretraining, it is input ids for language modeling task, which is randomly masked
                in finetuning, it is the unmasked input ids
            attentino_mask: attention mask for the backbone language model
            token_type_ids: segment ids
            node_ids: (batch_size, n_node)
            node_type_ids: (batch_size, n_node)
            node_scores: [bs, n_node, 1]
            adj_lengths: means the "actual" number of nodes (excluding padding) (batch_size, )
            adj -> edge_index, edge_type
                edge_index: list of (batch_size, );
                    each entry is torch.tensor(2, E(variable)) -> (2, total E)
                edge_type: list of (batch_size, );
                    each entry is torch.tensor(E(variable), ) -> (total E, )

        returns:
        last_hidden_state, hidden_states, attentions
        """
        assert len(input_ids.shape) == 2, "lm_input_ids should be [batch_size, seq_len]"

        edge_index, edge_type, pos_triples, neg_nodes = self.batch_graph(edge_index, edge_type, pos_triples, neg_nodes,
                                                                         node_ids.size(1))
        device = node_type_ids.device
        adj     = (edge_index.to(device), edge_type.to(device))
        lp_data = (pos_triples.to(device), neg_nodes.to(device))

        lm_outputs, gnn_output, pooled_represenation, link_losses= self.lmgnn(
            input_ids=input_ids,
            attention_mask=attention_mask, # If we extend to more than mlm, this needs to match lm_input_ids
            token_type_ids=token_type_ids,
            output_mask=output_mask,
            node_ids=node_ids,
            node_type_ids=node_type_ids,
            node_scores=node_scores,
            adj_lengths=adj_lengths,
            special_nodes_mask=special_nodes_mask,
            adj=adj,
            lp_data=lp_data,
            emb_data=None,
            inputs_embeds=inputs_embeds,)

        last_hidden_state, hidden_states, attentions = lm_outputs.values()
        pooled_language_representation, pooled_gnn_representation = pooled_represenation
        if not return_dict:
            outputs = (last_hidden_state, )
            if output_hidden_states:
                outputs += (hidden_states, )
            if output_attentions:
                outputs += (attentions, )
            outputs += (pooled_language_representation, pooled_gnn_representation, link_losses, )
            return outputs
        return DragonEncoderOutput(last_hidden_state=last_hidden_state,
                                   hidden_states=hidden_states,
                                   attentions=attentions,
                                   pooled_language_representation=pooled_language_representation,
                                   pooled_gnn_representation=pooled_gnn_representation,
                                   link_losses=link_losses,
                                   gnn_hidden_states=gnn_output)

    def get_input_embeddings(self):
        """Returns the model's input embeddings."""
        return self.lmgnn.backbone.get_input_embeddings()

    def get_output_embeddings(self):
        """Returns the model's output embeddings."""
        return self.lmgnn.backbone.get_output_embeddings()

    def freeze_lm(self):
        """Freeze weights of the language model.
        The node embedding is always frozen.
        """
        # The first two are for t5 backbone, the last is for the node embedding
        freeze_patterns= ['encoder.block', 'encoder.final_layer_norm', 'shared.weight', 'node_emb.emb.weight']
        for name, param in self.named_parameters():
            for freeze_pattern in freeze_patterns:
                if freeze_pattern in name:
                    param.requires_grad = False
                    break

    def unfreeze_lm(self):
        """Unfreeze weights of the language model.
        The node embedding is always frozen.
        """
        unfreeze_patterns= ['encoder.block', 'encoder.final_layer_norm', 'shared.weight']
        for name, param in self.named_parameters():
            for unfreeze_pattern in unfreeze_patterns:
                if unfreeze_pattern in name:
                    param.requires_grad = True
                    break

        for name, param in self.named_parameters():
            if 'node_emb.emb.weight' in name:
                param.requires_grad = False

    @classmethod
    def from_pretrained(cls, args):
        model = construct_encoder(args, model_cls=T5DragonEncoder)
        return model


class T5GNN(nn.Module):
    def __init__(self, config, args, k, n_ntype, n_etype,
                 n_node, node_dim, node_in_dim, n_attention_head,
                 fc_dim, n_fc_layer, p_emb, p_gnn, p_fc, pretrained_node_emb,
                 freeze_ent_emb, init_range, ie_dim, info_exchange, ie_layer_num,
                 sep_ie_layers, layer_id):
        '''
        k: the number of fusion layers
        '''
        super().__init__()
        self.args = args
        self.config = config

        self.init_range = init_range

        self.k = k
        self.node_dim = node_dim
        self.n_attention_head = n_attention_head
        self.activation = nn.GELU()
        if k >= 0:
            self.node_emb = CustomizedEmbedding(node_num=n_node, node_out_dim=node_dim, use_contextualized=False, node_in_dim=node_in_dim, pretrained_node_emb=pretrained_node_emb, freeze_ent_emb=freeze_ent_emb)
            self.pooler = MultiheadAttPoolLayer(n_attention_head, config.hidden_size, node_dim)

        concat_vec_dim = node_dim * 2 + config.hidden_size if k>=0 else config.hidden_size
        self.fc = MLP(concat_vec_dim, fc_dim, 1, n_fc_layer, p_fc, layer_norm=True)

        self.dropout_e = nn.Dropout(p_emb)
        self.dropout_fc = nn.Dropout(p_fc)

        if init_range > 0:
            self.apply(self._init_weights)

        self.backbone = TextKGMessagePassing(config, args=args, k=k, n_ntype=n_ntype, n_etype=n_etype, dropout=p_gnn, node_dim=node_dim, ie_dim=ie_dim, p_fc=p_fc, info_exchange=info_exchange, ie_layer_num=ie_layer_num, sep_ie_layers=sep_ie_layers) #this is equivalent to BertModel

        self.layer_id = layer_id
        self.cpnet_vocab_size = n_node

        if args.link_task:
            if args.link_decoder == 'DistMult':
                self.linkpred = DistMultDecoder(args, num_rels=n_etype, h_dim=node_dim)
            elif args.link_decoder == 'TransE':
                self.linkpred = TransEDecoder(args, num_rels=n_etype, h_dim=node_dim)
            elif args.link_decoder == 'RotatE':
                self.linkpred = RotatEDecoder(args, num_rels=n_etype, h_dim=node_dim)
            else:
                raise NotImplementedError()
            if args.link_proj_headtail:
                self.linkpred_proj = nn.Linear(node_dim, node_dim)
            if args.link_normalize_headtail == 3:
                self.emb_LayerNorm = nn.LayerNorm(node_dim)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                output_mask,
                node_ids,
                node_type_ids,
                node_scores,
                adj_lengths,
                special_nodes_mask,
                adj,
                lp_data,
                emb_data=None,
                inputs_embeds=None):
        """
        input_ids: in the pretraining time, this is corrupted
        node_ids: (batch_size, n_node)
        adj_lengths: (batch_size,)
        node_type_ids: (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_scores: (batch_size, n_node, 1)
        adj: edge_index, edge_type
        lp_data: pos_triples, neg_nodes

        returns:
        logits: [bs]
        """
        # GNN inputs
        node_ids[node_ids == 0] = self.cpnet_vocab_size + 2
        if self.k >= 0:
            gnn_input = self.node_emb(node_ids - 1, emb_data).to(node_type_ids.device)
        else:
            gnn_input = torch.zeros((node_ids.size(0), node_ids.size(1), self.node_dim)).float().to(node_type_ids.device)
        gnn_input[:, 0] = 0
        gnn_input = self.dropout_e(gnn_input) #(batch_size, n_node, dim_node)

        #Normalize node sore (use norm from Z)
        if self.args.no_node_score:
            node_scores = node_scores.new_zeros(node_scores.size())
        else:
            _mask = (torch.arange(node_scores.size(1), device=node_scores.device) < adj_lengths.unsqueeze(1)).float() #0 means masked out #[batch_size, n_node]
            node_scores = -node_scores
            node_scores = node_scores - node_scores[:, 0:1, :] #[batch_size, n_node, 1]
            node_scores = node_scores.squeeze(2) #[batch_size, n_node]
            node_scores = node_scores * _mask
            mean_norm  = (torch.abs(node_scores)).sum(dim=1) / adj_lengths  #[batch_size, ]
            node_scores = node_scores / (mean_norm.unsqueeze(1) + 1e-05) #[batch_size, n_node]
            node_scores = node_scores.unsqueeze(2) #[batch_size, n_node, 1]

        # Merged core
        lm_outputs, gnn_output = self.backbone(input_ids=input_ids,
                                               attention_mask=attention_mask,
                                               H=gnn_input,
                                               A=adj,
                                               node_type=node_type_ids,
                                               node_score=node_scores,
                                               special_nodes_mask=special_nodes_mask,
                                               output_hidden_states=True,
                                               output_attentions=True,
                                               inputs_embeds=inputs_embeds)
        # lm_outputs: BaseModelOutput
        # gnn_output: [bs, n_node, dim_node]

        # LM outputs
        all_hidden_states = lm_outputs.hidden_states # ([bs, seq_len, sent_dim] for _ in range(25))
        lm_hidden_states = all_hidden_states[self.layer_id] # [bs, seq_len, sent_dim]
        sent_vecs = lm_hidden_states[:, 0] # simply take the hidden states corresponding to the first token

        # GNN outputs
        Z_vecs = gnn_output[:,0]   #(batch_size, dim_node)

        node_mask = torch.arange(node_type_ids.size(1), device=node_type_ids.device) >= adj_lengths.unsqueeze(1) #[bs, nodes] 1 means masked out
        gnn_output = gnn_output * (~node_mask).float().unsqueeze(2)
        node_mask = node_mask | (node_type_ids == 3) # pool over all KG nodes (excluding the context node)
        node_mask[node_mask.all(1), 0] = 0  # a temporary solution to avoid zero node

        if self.args.link_task:
            pos_triples, neg_nodes = lp_data #pos_triples: [3, `total_n_triple`],  neg_nodes: [`total_n_triple`, n_neg]

            pos_samples = pos_triples #[3, `total_n_triple`]

            _n_neg = neg_nodes.size(1)
            head_negative_sample = neg_nodes[:, :_n_neg//2]             #[`total_n_triple`, n_neg//2]
            tail_negative_sample = neg_nodes[:, _n_neg//2:_n_neg//2*2]  #[`total_n_triple`, n_neg//2]

            _, _, gnn_dim = gnn_output.size()
            embs = gnn_output.view(-1, gnn_dim) #[`total_n_nodes`, gnn_dim]

            if self.args.link_proj_headtail:
                embs = self.linkpred_proj(embs)
            if self.args.link_normalize_headtail == 1:
                embs = embs / torch.norm(embs, p=2, dim=1, keepdim=True).detach()
            elif self.args.link_normalize_headtail == 2:
                embs = torch.tanh(embs)
            elif self.args.link_normalize_headtail == 3:
                embs = self.emb_LayerNorm(embs)

            positive_score  = self.linkpred(embs, pos_samples) #[`total_n_triple`, 1]
            head_neg_scores = self.linkpred(embs, (pos_samples, head_negative_sample), mode='head-batch')
            tail_neg_scores = self.linkpred(embs, (pos_samples, tail_negative_sample), mode='tail-batch')
            negative_score = torch.cat([head_neg_scores, tail_neg_scores], dim=-1) #[`total_n_triple`, total_n_neg]
            scores = (positive_score, negative_score)

            link_loss, pos_link_loss, neg_link_loss = self.linkpred.loss(scores)
        else:
            link_loss, pos_link_loss, neg_link_loss = 0, 0, 0
        
        # yc: return lm_hidden_states and gnn_output if return_hidden_states is set
        # lm_hidden_states: [bs, seq_len, sent_dim], gnn_output:  [bs, n_node, dim_node]
        pooled_represenation = (sent_vecs, Z_vecs) # [bs, sent_dim], [bs, dim_node]
        return lm_outputs, gnn_output, pooled_represenation, (link_loss, pos_link_loss, neg_link_loss)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        lmgnn_kwargs = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(T5GNN.__init__).args}
        print(f"Left out args from lmgnn: {set(kwargs.keys()) - set(lmgnn_kwargs.keys())}")
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        model = T5GNN(config, **lmgnn_kwargs)
        tkm_kwargs = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(TextKGMessagePassing.__init__).args}
        print(f"Left out args from TextKGMessagePassing: {set(kwargs.keys()) - set(tkm_kwargs.keys())}")
        text_message_passing, loading_args = TextKGMessagePassing.from_pretrained(pretrained_model_name_or_path, *inputs,
                                                                                    output_loading_info=True,
                                                                                    **tkm_kwargs) #this is equivalent to BertModel
        model.backbone = text_message_passing
        return model, loading_args


class TextKGMessagePassing(T5EncoderModel):
    """In encoder-only architectures, the model inherits e.g. BertModel(embedding, encoder, ..),
    with the encoder is replaced by a GNN encoder, which inherits e.g. BertEncoder.
    When adapting to encoder-decoder architectures, the model inherits e.g. BartModel(embedding, encoder, decoder, ..),
    with the encoder is replaced by a GNN encoder, which inherits e.g. BartEncoder.
    """
    def __init__(self, config, args, k, n_ntype, n_etype, dropout=0.2, node_dim=200, ie_dim=200, p_fc=0.2, info_exchange=True, ie_layer_num=1, sep_ie_layers=False):
        '''
        k: the number of fusion layers
        n_ntype: number of node types
        n_etype: number of edge types
        cocnept_dim: dimension of node embedding
        ie_dim: dimension of information exchange???
        '''
        super().__init__(config=config)

        self.n_ntype = n_ntype
        self.n_etype = n_etype

        self.hidden_size = node_dim
        self.emb_node_type = nn.Linear(self.n_ntype, node_dim // 2)

        self.basis_f = 'sin' #['id', 'linact', 'sin', 'none']
        if self.basis_f in ['id']:
            self.emb_score = nn.Linear(1, node_dim // 2)
        elif self.basis_f in ['linact']:
            self.B_lin = nn.Linear(1, node_dim // 2)
            self.emb_score = nn.Linear(node_dim // 2, node_dim // 2)
        elif self.basis_f in ['sin']:
            self.emb_score = nn.Linear(node_dim // 2, node_dim // 2)

        self.k = k

        self.Vh = nn.Linear(node_dim, node_dim)
        self.Vx = nn.Linear(node_dim, node_dim)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout


        embed_tokens = self.shared
        self.encoder = T5GAT(config, args, k=k, n_ntype=n_ntype, n_etype=n_etype,
                             hidden_size=node_dim, dropout=dropout,
                             node_dim=node_dim, ie_dim=ie_dim, p_fc=p_fc,
                             info_exchange=info_exchange, ie_layer_num=ie_layer_num,
                             sep_ie_layers=sep_ie_layers, embed_tokens=embed_tokens)

        self.sent_dim = config.hidden_size

    def forward(self, input_ids, attention_mask, H, A, node_type, node_score, special_nodes_mask,
                cache_output=False, position_ids=None, head_mask=None, output_hidden_states=True, output_attentions=False,
                inputs_embeds=None):
        """
        input_ids: [bs, seq_len]
        token_type_ids: [bs, seq_len]
        attention_mask: [bs, seq_len]
        H: tensor of shape (batch_size, n_node, d_node)
            node features from the previous layer
        A: (edge_index, edge_type)
            edge_index: [2, n_edges]
            edge_type: [n_edges]
        node_type: long tensor of shape (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_score: tensor of shape (batch_size, n_node, 1)
        
        Returns:
            outputs: (sequence_output, pooled_output, (hidden_states), (attentions))
            output: tensor of shape ___
        """
        # GNN inputs
        _batch_size, _n_nodes = node_type.size()

        #Embed type
        T = make_one_hot(node_type.view(-1).contiguous(), self.n_ntype).view(_batch_size, _n_nodes, self.n_ntype)
        node_type_emb = self.activation(self.emb_node_type(T)) #[batch_size, n_node, dim/2]

        #Embed score
        if self.basis_f == 'sin':
            js = torch.arange(self.hidden_size//2).unsqueeze(0).unsqueeze(0).float().to(node_type.device) #[1,1,dim/2]
            js = torch.pow(1.1, js) #[1,1,dim/2]
            B = torch.sin(js * node_score) #[batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]
        elif self.basis_f == 'id':
            B = node_score
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]
        elif self.basis_f == 'linact':
            B = self.activation(self.B_lin(node_score)) #[batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]

        X = H
        edge_index, edge_type = A #edge_index: [2, total_E]   edge_type: [total_E, ]  where total_E is for the batched graph
        _X = X.view(-1, X.size(2)).contiguous() #[`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node
        _node_type = node_type.view(-1).contiguous() #[`total_n_nodes`, ]
        _node_feature_extra = torch.cat([node_type_emb, node_score_emb], dim=2).view(_node_type.size(0), -1).contiguous() #[`total_n_nodes`, dim]

        # Merged core
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            _X=_X,
            edge_index=edge_index,
            edge_type=edge_type,
            _node_type=_node_type,
            _node_feature_extra=_node_feature_extra,
            special_nodes_mask=special_nodes_mask,
            return_dict=True
        )

        # LM outputs
        sequence_output = encoder_outputs.last_hidden_state
        lm_outputs = BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions
        )

        # GNN outputs
        _X = encoder_outputs.node_features
        X = _X.view(node_type.size(0), node_type.size(1), -1) #[batch_size, n_node, dim]

        gnn_output = self.activation(self.Vh(H) + self.Vx(X))
        gnn_output = self.dropout(gnn_output)

        return lm_outputs, gnn_output


class T5GAT(T5Stack):
    """The encoder model in TextKGMessagePassing."""
    def __init__(self, config, args, k, n_ntype, n_etype, hidden_size=200, dropout=0.2,
                 node_dim=200, ie_dim=200, p_fc=0.2, info_exchange=True, ie_layer_num=1,
                 sep_ie_layers=False, embed_tokens=None):

        # init with T5Stack
        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        super().__init__(encoder_config, embed_tokens)

        self.args = args
        self.k = k
        self.node_dim = node_dim
        self.num_hidden_layers = config.num_hidden_layers
        self.info_exchange = info_exchange
        if k >= 1:
            self.edge_encoder = torch.nn.Sequential(torch.nn.Linear(n_etype + 1 + n_ntype * 2, hidden_size), torch.nn.BatchNorm1d(hidden_size), torch.nn.ReLU(), torch.nn.Linear(hidden_size, hidden_size))
            self.gnn_layers = nn.ModuleList([GATConvE(args, hidden_size, n_ntype, n_etype, self.edge_encoder) for _ in range(k)])
            self.activation = nn.GELU()
            self.dropout_rate = dropout

            self.sent_dim = config.hidden_size
            self.sep_ie_layers = sep_ie_layers
            if sep_ie_layers:
                self.ie_layers = nn.ModuleList([MLP(self.sent_dim + node_dim, ie_dim, self.sent_dim + node_dim, ie_layer_num, p_fc) for _ in range(k)])
            else:
                self.ie_layer = MLP(self.sent_dim + node_dim, ie_dim, self.sent_dim + node_dim, ie_layer_num, p_fc)
            if self.args.residual_ie == 2:
                self.ie_LayerNorm = nn.LayerNorm(self.sent_dim + node_dim)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        *,
        _X=None,
        edge_index=None,
        edge_type=None,
        _node_type=None,
        _node_feature_extra=None,
        special_nodes_mask=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=None):
        """
        hidden_states: [bs, seq_len, sent_dim]
        attention_mask: [bs, 1, 1, seq_len]
        head_mask: list of shape [num_hidden_layers]

        _X: [`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node
        edge_index: [2, n_edges]
        edge_type: [n_edges]
        _node_type: [bs * n_nodes]
        _node_feature_extra: [bs * n_nodes, node_dim]
        """

        # Copied from T5Stack.forward -- START
        # decoder-related snippets are removed
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                f"You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(f"You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            # assert (torch.max(input_ids) < self.embed_tokens.weight.size(0)).item()
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=inputs_embeds.device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        position_bias = None

        hidden_states = self.dropout(inputs_embeds)
        # Copied from T5Stack.forward -- END

        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.block):
            # LM
            layer_head_mask = head_mask[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states,
                                         attention_mask=extended_attention_mask,
                                         position_bias=position_bias,
                                         layer_head_mask=layer_head_mask,
                                         use_cache=use_cache,
                                         output_attentions=output_attentions,
                                         return_dict=True)
            hidden_states = layer_outputs[0]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = None

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[-1],)

            if i >= self.num_hidden_layers - self.k:
                # GNN
                gnn_layer_index = i - self.num_hidden_layers + self.k
                _X = self.gnn_layers[gnn_layer_index](_X, edge_index, edge_type, _node_type, _node_feature_extra)
                _X = self.activation(_X)
                _X = F.dropout(_X, self.dropout_rate, training = self.training)

                # Exchange info between LM and GNN hidden states (Modality interaction)
                if self.info_exchange == True or (self.info_exchange == "every-other-layer" and (i - self.num_hidden_layers + self.k) % 2 == 0):
                    X = _X.view(batch_size, -1, _X.size(1)) # [bs, max_num_nodes, node_dim]
                    context_node_lm_feats = hidden_states[:, 0, :] # [bs, sent_dim]
                    context_node_gnn_feats = X[:, 0, :] # [bs, node_dim]
                    context_node_feats = torch.cat([context_node_lm_feats, context_node_gnn_feats], dim=1)
                    if self.sep_ie_layers:
                        _context_node_feats = self.ie_layers[gnn_layer_index](context_node_feats)
                    else:
                        _context_node_feats = self.ie_layer(context_node_feats)
                    if self.args.residual_ie == 1:
                        context_node_feats = context_node_feats + _context_node_feats
                    elif self.args.residual_ie == 2:
                        context_node_feats = self.ie_LayerNorm(context_node_feats + _context_node_feats)
                    else:
                        context_node_feats = _context_node_feats
                    context_node_lm_feats, context_node_gnn_feats = torch.split(context_node_feats, [context_node_lm_feats.size(1), context_node_gnn_feats.size(1)], dim=1)
                    # Stop the information from flowing to the LM layers
                    # hidden_states[:, 0, :] = context_node_lm_feats
                    X[:, 0, :] = context_node_gnn_feats
                    _X = X.view_as(_X)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            outputs = (hidden_states,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_attentions,)
            return outputs + (_X,) # last-layer hidden state, (all hidden states), (all attentions)
        return T5GATOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            node_features=_X
        )
