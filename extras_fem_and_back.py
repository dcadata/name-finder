import pandas as pd
import seaborn as sns
from matplotlib import pyplot as mpl

from extras import PLACEHOLDER_NAMES


def create_fem_and_back_analysis(calcd: pd.DataFrame) -> pd.DataFrame:
    number_min_total_cutoff = 1_000
    number_min_per_half_decade = 25
    ratio_f_cutoff = 0.25
    swing_back_cutoff = -0.1
    start_year_cutoff = 1950
    end_year_cutoff = 2020

    totals = calcd[calcd.year >= start_year_cutoff].groupby('name', as_index=False).number.sum()
    totals = totals[totals.number > number_min_total_cutoff]

    df = calcd[(calcd.year >= start_year_cutoff) & (calcd.year < end_year_cutoff) & calcd.name.isin(totals.name)].copy()
    df['half_decade'] = df.year.apply(lambda x: f'{str(x)[:3]}0_' + ('1' if int(str(x)[-1]) >= 5 else '0'))
    df = df.groupby(['name', 'half_decade'], as_index=False)[['number', 'number_f']].sum()
    df = df[df.number >= number_min_per_half_decade].copy()
    df['ratio_f'] = df.number_f / df.number

    latests = df.drop_duplicates(subset=['name'], keep='last')

    grouped = df.groupby('name', as_index=False).agg(dict(ratio_f=max, number=sum))
    df = grouped.merge(latests[['name', 'ratio_f']], on='name', suffixes=('_max', '_latest'))
    df['swing_back'] = df.ratio_f_latest - df.ratio_f_max

    df = df[
        (df.ratio_f_max > ratio_f_cutoff) &
        (df.ratio_f_latest < (1 - ratio_f_cutoff)) &
        (df.swing_back < swing_back_cutoff) &
        ~df.name.isin(PLACEHOLDER_NAMES)
        ]
    df = df.sort_values('number', ascending=False)[['name', 'ratio_f_max', 'ratio_f_latest', 'swing_back']]

    df = df.iloc[:150].copy()
    for col in ('ratio_f_max', 'ratio_f_latest', 'swing_back'):
        df[col] = (df[col].round(2) * 100).map(int)

    with open('extras_outputs/fem_and_back.txt', 'w') as file:
        file.write(df.to_markdown(index=False))

    return df


def create_fem_and_back_visualization(fem_and_back: pd.DataFrame) -> None:
    figsize = (5, 24)
    sns.set_theme(style='whitegrid')

    f, ax = mpl.subplots(figsize=figsize)

    sns.barplot(data=fem_and_back, x='ratio_f_max', y='name', label='most fem %', color='orange')
    sns.barplot(data=fem_and_back, x='ratio_f_latest', y='name', label='latest fem %', color='purple')

    ax.set_title('some names that went fem and went back')
    ax.set(xlim=(0, 100), ylabel='', xlabel='fem %')
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.tick_params(axis='y', labelsize=10)
    ax.legend(ncol=2, loc='upper right', frameon=True)
    sns.despine(left=True, bottom=False)

    ax.figure.set_size_inches(*figsize)
    ax.figure.tight_layout()

    ax.figure.savefig('extras_outputs/fem_and_back.png')
