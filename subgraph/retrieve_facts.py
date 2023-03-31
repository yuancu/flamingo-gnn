"""Retrieve all one hop facts from a given set of entities.
We use CLOCQ's endpoint service to achieve this: https://clocq.mpi-inf.mpg.de
It brutally go through each linked entity in each question, retrieving one-hop facts for each entity.
"""
import argparse
import requests
import json
from functools import lru_cache
from pathlib import Path

from tqdm import tqdm
import srsly

DEFAULT_ENDPOINT = 'https://clocq.mpi-inf.mpg.de/api/neighborhood'

@lru_cache
def get_one_hop_facts(endpoint, qid):
    url = f'{endpoint}?item={qid}&p=10'

    response = requests.get(url, timeout=60)

    if response.status_code == 200: # check if the request was successful
        data = json.loads(response.content.decode('utf-8')) # parse the JSON response
        return data
    else:
        print('Error with', qid, response.status_code, response.reason)
        return None
    
def main(args):
    # Each item is a dictionary with keys: 'id', 'qid', 'span', 'entity', where the qid is a list of qids 
    # related to the question.
    el_results = srsly.read_jsonl(args.el_path)
    with open(args.el_path, "r", encoding="utf-8") as f:
        total_lines = len(f.readlines())
    retrieval_results = []
    for el_result in tqdm(el_results, desc=f'Retrieving facts for {args.el_path}', total=total_lines):
        qids = el_result['qid']
        facts = []
        for qid in qids:
            one_hop_facts = get_one_hop_facts(args.endpoint, qid)
            if one_hop_facts is not None:
                facts = facts + one_hop_facts
        retrieval_result = {
            'id': el_result['id'],
            'facts': facts,
        }
        # Potential bottleneck: all results are stored in memory.
        retrieval_results.append(retrieval_result)
    parent_path = Path(args.output_path).parent
    if not parent_path.exists():
        parent_path.mkdir(parents=True)
        print(f'Created folder {parent_path} for {args.output_path}')
    srsly.write_jsonl(args.output_path, retrieval_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', type=str, default=DEFAULT_ENDPOINT, help='CLOCQ endpoint')
    parser.add_argument('--el-path', type=str, required=True, help='Path to the entity linked file')
    parser.add_argument('--output-path', type=str, required=True, help='Path to the output file')
    args = parser.parse_args()

    main(args)
