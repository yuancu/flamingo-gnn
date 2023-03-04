from argparse import Namespace

import yaml


def load_args(config_path='configs/lmgnn.yaml', profile='pretrain_squad'):
    with open(config_path, encoding='utf-8') as f:
        args = yaml.load(f, Loader=yaml.Loader)[profile]
    args = Namespace(**args['model'], **args['data'], **args['task'], **args['optim'],
                     **args['misc'])
    return args
