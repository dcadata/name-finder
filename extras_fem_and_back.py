import pandas as pd
import seaborn as sns
from matplotlib import pyplot as mpl

from extras import PLACEHOLDER_NAMES


def create_fem_and_back_analysis(calcd: pd.DataFrame) -> pd.DataFrame:
    number_min_total_cutoff = 10_000
    number_min_per_decade = 25
    ratio_f_cutoff = 0.3
    swing_back_cutoff = -0.1

    totals = calcd.groupby('name', as_index=False).number.sum()
    totals = totals[totals.number > number_min_total_cutoff]

    df = calcd[(calcd.year < 2020) & calcd.name.isin(totals.name)].copy()
    df['decade'] = df.year.apply(lambda x: f'{str(x)[:3]}0_' + ('1' if int(str(x)[-1]) >= 5 else '0'))
    df = df.groupby(['name', 'decade'], as_index=False)[['number', 'number_f']].sum()
    df = df[df.number >= number_min_per_decade].copy()
    df['ratio_f'] = df.number_f / df.number

    latests = df.drop_duplicates(subset=['name'], keep='last')

    grouped = df.groupby('name', as_index=False).agg(dict(ratio_f=max, number=sum))
    df = grouped.merge(latests[['name', 'ratio_f']], on='name', suffixes=('_max', '_latest'))
    df['swing_back'] = df.ratio_f_latest - df.ratio_f_max

    df = df[
        (df.ratio_f_max > ratio_f_cutoff) &
        (df.ratio_f_latest < (1 - ratio_f_cutoff)) &
        (df.ratio_f_latest > 0.01) &
        (df.swing_back < swing_back_cutoff) &
        ~df.name.isin(PLACEHOLDER_NAMES)
        ]
    df = df.sort_values('number', ascending=False)
    return df


def create_fem_and_back_visualization(fem_and_back: pd.DataFrame) -> None:
    sns.set_theme(style='whitegrid')

    f, ax = mpl.subplots(figsize=(2, 20))
    f.tight_layout()

    sns.barplot(data=fem_and_back, x='ratio_f_max', y='name', label='max', color='orange')
    sns.barplot(data=fem_and_back, x='ratio_f_latest', y='name', label='latest', color='purple')

    ax.set_title('some names that went fem and went back')
    ax.legend(ncol=2, loc='upper right', frameon=True)
    ax.set(xlim=(0, 1), ylabel='', xlabel='ratio_f')
    ax.tick_params(labelsize=10, axis='y')
    sns.despine(left=True, bottom=True)
