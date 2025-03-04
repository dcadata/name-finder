import pandas as pd

from core import Year, DFAgg, Displayer, _standardize_name


def _build_predict_gender_reference(
        after: int = Year.DATA_QUALITY_BEST_AFTER,
        before: int = None,
        ratio_min: float = .8,
        number_min: int = 25,
        displayer: Displayer = None,
) -> pd.DataFrame:
    df = displayer.calculated.copy()
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


def predict_gender_batch(data: list[dict], **kwargs) -> list[dict]:
    df = pd.DataFrame(data)
    if 'name' not in df.columns:
        return []
    df = df.dropna(subset=['name'])
    df['matched_name'] = df.name.astype(str).map(_standardize_name)

    reference = _build_predict_gender_reference(**kwargs)
    df = df.merge(reference.rename(columns=dict(name='matched_name')), on='matched_name', how='left')
    df.gender_prediction = df.gender_prediction.fillna('unk')
    return df.to_dict('records')


def _create_age_reference_for_mid_percentile(displayer: Displayer, mid_percentile: float) -> pd.DataFrame:
    lower_percentile = .5 - mid_percentile / 2
    upper_percentile = 1 - lower_percentile
    # noinspection PyProtectedMember
    age_reference: pd.DataFrame = displayer._age_reference.copy()
    id_cols = ['name', 'sex']

    df = age_reference[age_reference.year >= Year.DATA_QUALITY_BEST_AFTER].copy()
    df.number_living_pct = df.groupby(id_cols).number_living_pct.cumsum()
    df['lower'] = (lower_percentile - df.number_living_pct).abs()
    df['upper'] = (upper_percentile - df.number_living_pct).abs()

    lower_and_upper_mins = df.groupby(id_cols, as_index=False)[['lower', 'upper']].min()
    agg_cols = [*id_cols, 'lower']
    lowers = df.merge(lower_and_upper_mins[agg_cols], on=agg_cols)
    agg_cols = [*id_cols, 'upper']
    uppers = df.merge(lower_and_upper_mins[agg_cols], on=agg_cols)
    df = pd.concat((lowers, uppers)).rename(columns=dict(year='year_lower'))
    df['year_upper'] = df.year_lower.copy()
    df = df.groupby(id_cols, as_index=False).agg(dict(year_lower='min', year_upper='max'))
    return df


def predict_age_batch(displayer: Displayer, mid_percentile: float, data: list[dict[str, str]]) -> list[dict]:
    names = pd.DataFrame(data)
    if 'name' not in names.columns or 'sex' not in names.columns:
        return []
    names = names.dropna()
    names['matched_name'] = names.name.astype(str).map(_standardize_name)
    names['matched_sex'] = names.sex.astype(str).str.lower()

    df = _create_age_reference_for_mid_percentile(displayer, mid_percentile)
    df = names.merge(df, left_on=['matched_name', 'matched_sex'], right_on=['name', 'sex'], how='left', suffixes=(
        '', '_ref')).drop(columns=['name_ref', 'sex_ref'])
    return df.to_dict('records')
