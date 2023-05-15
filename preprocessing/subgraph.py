"""
Create subgraphs from grounded statements.

Input:
    grounded_statements.jsonl, where each line is a dict:
        id: str
        question: str
        question_entities: list[str]
        answers_entities: list[str]
        ...

Output: subgraph.jsonl, where each line is a subgraph dict:
    id: str
    entities: list[str]
    connections: list[tuple]
        where each tuple is (head_loc, relation_id, tail_loc),
        head and tail locs are the indices of the entities in the entities list.
        relation_id is the identifier of the relation
        e.g. (0, 'P31', 9)
    num_known_entities: int
        number of known entities in the entities list, those come after are unknown. 
"""
import numpy as np
import srsly
from tqdm import tqdm

from .wikidata import Wikidata


def create_subgraph(ground_path, output_path, src_key, dst_key, sparql_endpoint):
    """
    Create subgraphs from grounded statements.
    """
    statements = srsly.read_jsonl(ground_path)
    subgraphs = []
    kg = Graph(sparql_endpoint)
    for statement in tqdm(statements, desc=f"Creating subgraphs from {ground_path}"):
        triplets = get_connections(kg, statement[src_key], statement[dst_key])
        entities = set([triplet[0] for triplet in triplets] + [triplet[-1] for triplet in triplets])
        known_entities = list(entities + set(statement[src_key]) - statement[dst_key]) 
        num_known_entities = len(known_entities)
        entities = known_entities + statement[dst_key]
        entity_map = {entity: i for i, entity in enumerate(entities)}
        connections = [(entity_map[triplet[0]], triplet[1], entity_map[triplet[-1]]) for triplet in triplets]
        subgraph = {
            **statement,
            'entities': entities,
            'connections': connections,
            'num_known_entities': num_known_entities
        }
        subgraphs.append(subgraph)
    srsly.write_jsonl(output_path, subgraphs)


def get_connections(knowledge_graph, src_entities, dst_entities):
    """
    Get the connections between src_entities and dst_entities.
    """
    paths = []
    for src_entity in src_entities:
        for dst_entity in dst_entities:
            path = knowledge_graph.search_path(src_entity, dst_entity)
            paths.extend(path)
    return paths


class Graph:
    def __init__(self, wikidata_endpoint, known_relations=None) -> None:
        self.wikidata = Wikidata(wikidata_endpoint)
        self.known_relations = known_relations
    
    def search_path(self, src_entity, dst_entity):
        """
        Search for the path from src_entity to dst_entity in the graph.
        """
        one_hop_relations = self.wikidata.search_one_hop_relations(src_entity, dst_entity)
        if len(one_hop_relations) > 0:
            return [(src_entity, relation, dst_entity) for relation in one_hop_relations
                    if not self.known_relations or relation in self.known_relations]
        two_hop_relations = self.wikidata.search_two_hop_relations(src_entity, dst_entity)
        return [(src_entity, relation1, entity1, relation2, dst_entity) for relation1, entity1, relation2 in two_hop_relations
                if not self.known_relations or (relation1 in self.known_relations and relation2 in self.known_relations)]


