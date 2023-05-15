"""
Ground the statements to Wikidata entities.

Input: statements.jsonl, where each line is a statement dict:
    id: str
    question: str
    answers: list[str]
    context: str
    ...

Output: grounded.jsonl, where each line is a ground dict like:
    id: str
    question_entities: list[str]
    answers_entities: list[str]
    context_entities: list[str]
    ...
"""

import srsly
from tqdm import tqdm


def ground(text):
    """Ground the text to Wikidata entities.

    Args:
        text (str): Text to be grounded.
    """
    raise NotImplementedError


def ground_statement(statement_path, output, ground_on=['question'],
                     add_title_entity=False, title_to_qid_fn=None):
    """
    Ground the statements and save the grounded statements to output.
    For each key in ground_on, we stored the grounded entities to key + '_entities'.

    Args:
        split_names: a list of split names, e.g. ['train', 'validation']
        ground_on (list): the keys to ground on. 
        add_title_entity: whether to add title entity to the context
        title_to_qid_fn: a function that takes a title and returns a qid
    """
    # Sanity check
    assert ground_on in ['question', 'context']
    if add_title_entity:
        assert title_to_qid_fn is not None

    print("statement path:", statement_path)
    statements = srsly.read_jsonl(statement_path)
    grounded_statements = []
    for statement in tqdm(statements, desc=f"Grounding {statement_path}"):
        if add_title_entity:
            title_entity = title_to_qid_fn(statement['title'])
            statement['title_entity'] = title_entity
        for key in ground_on:
            text = statement[key]
            statement[key + '_entities'] = ground(text)
        grounded_statements.append(statement)
    srsly.write_jsonl(output, grounded_statements)
