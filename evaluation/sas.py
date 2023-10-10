from haystack.eval import EvalAnswers

# model_path = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
model_path = 'cross-encoder/stsb-roberta-large'
eval_reader = EvalAnswers(sas_model=model_path)

def compute_sas(predictions, answers):
    """
    Compute SAS score of the predictions against references.
    """
    score = eval_reader.eval(
        answers=answers,
        predictions=predictions,
        top_k=1,
        match_type="string",
        return_preds=True,
    )
    return {'sas': score['sas'] }

if __name__ == '__main__':
    predictions = [
        'Danish-Norwegian patronymic surname meaning ”son of Anders”',
    ]
    references = [
        ['Denmark']
    ]
    print(compute_sas(predictions, references))
