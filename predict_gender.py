import pandas as pd

from core import Year, Filepath, DFAgg, Displayer, _standardize_name


class FilepathCalcd(Filepath):
    CALCD: str = Filepath.DATA_DIR + 'generated/calcd.csv'


def _refresh_calcd() -> None:
    displayer = Displayer()
    displayer.build_base()
    displayer.calculated.to_csv(FilepathCalcd.CALCD, index=False)
    return


def _build_predict_gender_reference(
        after: int = Year.DATA_QUALITY_BEST_AFTER,
        before: int = None,
        ratio_min: float = .8,
        number_min: int = 25,
) -> pd.DataFrame:
    df = pd.read_csv(FilepathCalcd.CALCD)
    number_min = max(number_min, 25)  # shouldn't be less than 25

    if after:
        df = df[df.year >= after]
    if before:
        df = df[df.year <= before]

    df = df.groupby('name', as_index=False).agg(DFAgg.NUMBER_SUM)

    df.loc[df.number_f > df.number_m, 'gender_prediction'] = 'f'
    df.loc[df.number_f < df.number_m, 'gender_prediction'] = 'm'
    df.loc[df.number_f == df.number_m, 'gender_prediction'] = 'x'

    ratio_f = df.number_f / df.number
    ratio_m = df.number_m / df.number
    df.loc[(ratio_f < ratio_min) & (ratio_m < ratio_min), 'gender_prediction'] = 'x'

    df['f_pct'] = (ratio_f * 100).round().map(int)
    df['m_pct'] = (ratio_m * 100).round().map(int)

    df.loc[df.number < number_min, 'gender_prediction'] = 'rare'
    df.gender_prediction = df.gender_prediction.fillna('unk')
    return df[['name', 'gender_prediction', 'f_pct', 'm_pct']]


def process_batch(data: list[dict], **kwargs) -> list[dict]:
    df = pd.DataFrame(data)
    if 'name' not in df.columns:
        return []
    df = df.dropna(subset=['name'])
    df['matched_name'] = df.name.astype(str).map(_standardize_name)

    reference = _build_predict_gender_reference(**kwargs)
    df = df.merge(reference.rename(columns=dict(name='matched_name')), on='matched_name', how='left')
    df.gender_prediction = df.gender_prediction.fillna('unk')
    return df.to_dict('records')
