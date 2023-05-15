"""
The statement file is a jsonl file that contains the following fields:
    id: str
    context: str
    question: dict
    answers: list[str]
    title: str
"""
import os
from pathlib import Path

import srsly
from tqdm import tqdm


class StatementReader:
    """Create statement iterators for different datasets. The iterator should return
    a statement dict.
    """
    @classmethod
    def example_reader(cls):
        """An example reader that returns a statement dict. The None examples will be skipped.

        Yields:
            dict: a statement dict
        """
        for i in range(5):
            if i == 3:
                yield None
            else:
                yield {
                    'id': '',
                    'context': '',
                    'question': '',
                    'answers': [''],
                    'title': ''
                }


def create_statement(reader, save_path):
    """Create a statement file from a reader. The None examples will be skipped.

    Args:
        reader: an iterator that returns a statement dict
        save_path: the path to save the statement file
    """
    skipped = 0
    statements = []
    for example in tqdm(reader):
        if example is None:
            skipped += 1
        else:
            statements.append(example)
    print(f"len statement: {len(statements)}; skipped {skipped}")
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    srsly.write_jsonl(save_path, statements)
    print(f"Statement saved to {save_path}")


if __name__ == '__main__':
    example_reader = StatementReader.example_reader()
    create_statement(example_reader, 'example.statement.jsonl')
