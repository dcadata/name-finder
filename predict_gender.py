import pandas as pd

from core import Displayer, Filepath, build_predict_gender_reference, _standardize_name


def process_batch(data: list[dict], displayer: Displayer = None, after: int = None, before: int = None) -> list[dict]:
    df = pd.DataFrame(data)
    if 'name' not in df.columns:
        return []
    df = df.dropna(subset=['name'])
    df['matched_name'] = df.name.astype(str).map(_standardize_name)

    if displayer is None or (not after and not before):
        reference = pd.read_csv(Filepath.GENDER_PREDICTION_REFERENCE, dtype=str)
    else:
        reference = build_predict_gender_reference(displayer, after, before)

    df = df.merge(reference.rename(columns=dict(name='matched_name')), on='matched_name', how='left')
    df.gender_prediction = df.gender_prediction.fillna('unk')
    return df.to_dict('records')
