"""4. Get connectivity between each two grounded entities in the question.
Again, We use CLOCQ's endpoint service to achieve this: https://clocq.mpi-inf.mpg.de/api/connect

e.g.
python subgraph/get_connectivity.py --el-path data/squad/el/ft.train.jsonl\
    --output-path data/squad/connectivity/ft.train.jsonl
"""
import argparse
import requests
import json
from functools import lru_cache
from itertools import product

import srsly
import tqdm

DEFAULT_ENDPOINT = 'https://clocq.mpi-inf.mpg.de/api/connect'


@lru_cache
def get_paths(endpoint, qid1, qid2):
    """Get the shortest path between two entities.
    When there is no path between two entities, the response is None,
    else, it is a list of paths, each path is [qid, rel, qid].
    """
    url = f'{endpoint}?item1={qid1}&item2={qid2}'

    response = requests.get(url, timeout=60)

    if response.status_code == 200: # check if the request was successful
        data = json.loads(response.content.decode('utf-8')) # parse the JSON response
        return data
    else:
        print('Error with finding path between', qid1, 'and', qid2, ':', response.status_code, response.reason)
        return None


def main(args):
    el_results = srsly.read_jsonl(args.el_path)
    total_lines = sum(1 for _ in srsly.read_jsonl(args.el_path))
    connectivities = []
    for el_result in tqdm.tqdm(el_results, desc=f'Getting connectivity for {args.el_path}', total=total_lines):
        qids = el_result['qid']
        paths = []
        for qid1, qid2 in product(qids, qids):
            if qid1 != qid2:
                path = get_paths(args.endpoint, qid1, qid2)
                if path is not None:
                    paths.append(path)
        connectivity = {
            'id': el_result['id'],
            'paths': paths,
        }
        connectivities.append(connectivity)
    srsly.write_jsonl(args.output_path, connectivities)
    print(f'Connectivity saved to {args.output_path}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', type=str, default=DEFAULT_ENDPOINT, help='CLOCQ endpoint')
    parser.add_argument('--el-path', type=str, required=True, help='Path to the entity linked file')
    parser.add_argument('--output-path', type=str, required=True, help='Path to the output connectivity file')
    args = parser.parse_args()
    
    main(args)
