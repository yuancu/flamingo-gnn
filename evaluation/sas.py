import numpy as np
from haystack.modeling.evaluation.metrics import semantic_answer_similarity

model_path = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

def compute_sas(predictions, answers):
    """
    Compute SAS score of the predictions against references.
    """
    score = semantic_answer_similarity(
        gold_labels=answers,
        predictions=predictions,
        sas_model_name_or_path=model_path
    )
    top_1_sas = np.mean(score[0])
    return {'sas': top_1_sas }

if __name__ == '__main__':
    predictions = [
        'Danish-Norwegian patronymic surname meaning ”son of Anders”',
    ]
    references = [
        ['Denmark']
    ]
    print(compute_sas(predictions, references))
