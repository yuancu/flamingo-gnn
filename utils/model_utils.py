import torch
import numpy as np


# Utils for the encoder
def get_tweaked_num_relations(num_relations, cxt_node_connects_all):
    """
    Args:
        num_relations: the number of relations in the dataset, i.e. len(id2relation)
        ctx_node_connects_all: whether every other node is connected by a ctx node
    """
    # cxt2qlinked_rel and cxt2alinked_rel
    tweaked_num_relations = num_relations + 2
    if cxt_node_connects_all:
        tweaked_num_relations += 1
    return tweaked_num_relations


def construct_encoder(args, model_cls):
    """
    num_relation: the number of relations in the original KG: len(id2relation)
    final_num_relation: the number of relations in the final KG, e.g. len(id2relation) + 2
    model_type: 'dragon' or 'dragon_encoder'. 'dragon' is the model for pretraining;
        'dragon_encoder' is the model for downstream tasks, it is compatible with the
        EncoderDecoderModel in transformers.
    """

    num_relation = args.num_relations
    final_num_relation = get_tweaked_num_relations(
        num_relation, args.cxt_node_connects_all)

    # Load pretrained entity embeddings
    entity_emb = np.load(args.ent_emb_paths)
    print(f"Entity embedding (shape: {entity_emb.shape}) loaded from {args.ent_emb_paths}.")
    entity_emb = torch.tensor(entity_emb, dtype=torch.float)
    enity_num, entity_in_dim = entity_emb.size(0), entity_emb.size(1)
    print(f"| num_entities: {enity_num} |")
    if args.random_ent_emb:
        entity_emb = None
        freeze_ent_emb = False
        entity_in_dim = args.gnn_dim
    else:
        freeze_ent_emb = args.freeze_ent_emb

    n_ntype = 4
    n_etype = final_num_relation * 2
    print('final_num_relation', final_num_relation, 'len(id2relation)', num_relation)
    # It's alreay added once
    # if args.cxt_node_connects_all:
    #     n_etype += 2
    print('n_ntype', n_ntype, 'n_etype', n_etype)

    model = model_cls(args, args.encoder_name_or_path, k=args.k,
                n_ntype=n_ntype, n_etype=n_etype, n_concept=enity_num,
                concept_dim=args.gnn_dim, concept_in_dim=entity_in_dim,
                n_attention_head=args.att_head_num, fc_dim=args.fc_dim,
                n_fc_layer=args.fc_layer_num, p_emb=args.dropouti,
                p_gnn=args.dropoutg, p_fc=args.dropoutf, pretrained_concept_emb=entity_emb,
                freeze_ent_emb=freeze_ent_emb, init_range=args.init_range,
                ie_dim=args.ie_dim, info_exchange=args.info_exchange,
                ie_layer_num=args.ie_layer_num, sep_ie_layers=args.sep_ie_layers,
                layer_id=args.encoder_layer, no_lm_head=True)
    return model
