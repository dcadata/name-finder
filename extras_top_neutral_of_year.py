import pandas as pd

from core import Displayer
from extras import PLACEHOLDER_NAMES, NEUTRAL_RATIO_RANGE


def top_neutral_of_2022(displayer: Displayer) -> pd.DataFrame:
    def _transform(df: pd.DataFrame) -> pd.DataFrame:
        for col in ('number', 'number_f', 'number_m'):
            df[col] = df[col].apply(lambda x: f'{x:,}')
        for col in 'fm':
            df[f'percent_{col}'] = df[f'ratio_{col}'].apply(lambda x: f'{int(x * 100)}%')
        return df

    target_year = 2022
    after_year = target_year - 20
    top = 100

    target_year_data = _transform(displayer.search(year=target_year, gender=NEUTRAL_RATIO_RANGE, top=top))
    target_year_data = target_year_data[~target_year_data.name.isin(PLACEHOLDER_NAMES)].copy()
    after_year_data = _transform(displayer.search(year=after_year, top=25_000))
    interval_data = _transform(displayer.search(after=after_year, top=25_000)).drop(columns=['ratio_f', 'ratio_m'])

    merged = (
        target_year_data
        .merge(after_year_data, on='name', suffixes=('', f'_{after_year}'))
        .merge(interval_data, on='name', suffixes=('', f'_after{after_year}'))
    )

    merged.name = merged.name.apply(lambda x: f"'{x}")
    merged['percent_change'] = (merged.ratio_f - merged[f'ratio_f_{after_year}']).apply(lambda x: (
            ('f' if x > 0 else 'm') + f'+{int(abs(x) * 100)}%'
    ))
    drop_cols = list(filter(lambda x: x.startswith('ratio_') or x.endswith(f'_{after_year}'), merged.columns))
    merged = merged.drop(columns=drop_cols)
    return merged
