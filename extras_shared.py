import pandas as pd

from demos import SsaSex


def melt_applicants_data(apps: pd.DataFrame) -> pd.DataFrame:
    apps_melted = apps.melt(['year'], ['number_m', 'number_f', 'number'], 'sex', 'number_')
    apps_melted = apps_melted.rename(columns=dict(number_='number'))
    apps_melted.loc[apps_melted.sex == 'number_f', 'sex'] = SsaSex.Female
    apps_melted.loc[apps_melted.sex == 'number_m', 'sex'] = SsaSex.Male
    apps_melted.loc[apps_melted.sex == 'number', 'sex'] = SsaSex.All
    return apps_melted


def add_ratios(calcd: pd.DataFrame) -> pd.DataFrame:
    for s in SsaSex.Both:
        calcd[f'ratio_{s}'] = calcd[f'number_{s}'] / calcd.number
    return calcd


def offset_plot_year_by_sex(df: pd.DataFrame, year_field: str) -> pd.DataFrame:
    df[year_field] = df[year_field].map(float)
    df.loc[df.sex == 'f', year_field] -= .25
    df.loc[df.sex == 'm', year_field] += .25
    return df


def convert_year_to_decade_or_half_decade(series: pd.Series, half: bool = False) -> pd.Series:
    year_str_acc = series.map(str).str
    decade = year_str_acc.slice(0, 3)
    if half:
        half_decade = year_str_acc.slice(-1).map(int).apply(lambda x: 0 if x < 5 else 5).map(str)
        series = decade + half_decade
    else:
        series = decade + '0'
    return series.map(int)


def rerank_by_decade_or_half_decade(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby(['name', 'year', 'sex'], as_index=False).number.sum()
    df['rank_'] = df.groupby(['year', 'sex']).number.rank(method='min', ascending=False)
    return df
