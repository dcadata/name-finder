import pandas as pd

from core import Year, UnknownName, Displayer

_GENDER_CATEGORY_AFTER: int = 1960


def build_gender_ratio_after_year(raw: pd.DataFrame) -> pd.DataFrame:
    years = list(range(_GENDER_CATEGORY_AFTER, Year.MAX_YEAR + 1, 10))
    ratios = pd.concat(raw.loc[raw.year >= year, ['name', 'sex', 'number']].assign(after=year) for year in years)

    totals_by_name = ratios.groupby(['name', 'after'], as_index=False).number.sum()

    ratios = ratios[ratios.sex == 'f'].drop(columns='sex').groupby(['name', 'after'], as_index=False).sum().merge(
        totals_by_name, on=['name', 'after'], suffixes=('', '_total'), how='right')
    ratios.number = ratios.number.fillna(0)

    ratios['gender'] = ''
    ratio = ratios.number / ratios.number_total

    ratios.loc[ratio > .7, 'gender'] = 'NeutFem'
    ratios.loc[ratio > .9, 'gender'] = 'Fem'
    ratios.loc[ratio < .3, 'gender'] = 'NeutMasc'
    ratios.loc[ratio < .1, 'gender'] = 'Masc'
    ratios.loc[(ratio >= .3) & (ratio <= .7), 'gender'] = 'Neut'

    def _make_sorted_string(x) -> str:
        x = list(set(x))
        x.sort()
        return ', '.join(x)

    df = ratios.groupby('name', as_index=False).agg(dict(gender=_make_sorted_string))
    return df


def build_age_percentile_reference(age_reference: pd.DataFrame, mid_percentile: float) -> pd.DataFrame:
    lower_percentile = .5 - mid_percentile / 2
    upper_percentile = 1 - lower_percentile

    df = age_reference[age_reference.year >= Year.DATA_QUALITY_BEST_AFTER].copy()
    id_cols = ['name', 'sex']

    df.number_living_pct = df.groupby(id_cols).number_living_pct.cumsum()
    df['lower'] = (lower_percentile - df.number_living_pct).abs()
    df['upper'] = (upper_percentile - df.number_living_pct).abs()

    lower_and_upper_mins = df.groupby(id_cols, as_index=False)[['lower', 'upper']].min()
    agg_cols = [*id_cols, 'lower']
    lowers = df.merge(lower_and_upper_mins[agg_cols], on=agg_cols)
    agg_cols = [*id_cols, 'upper']
    uppers = df.merge(lower_and_upper_mins[agg_cols], on=agg_cols)
    df = pd.concat((lowers, uppers)).rename(columns=dict(year='middle_lo'))
    df['middle_hi'] = df.middle_lo.copy()
    df = df.groupby(['name', 'sex'], as_index=False).agg(dict(middle_lo='min', middle_hi='max'))

    df = df[df.sex == 'f'].drop(columns='sex').merge(
        df[df.sex == 'm'].drop(columns='sex'), on='name', how='outer', suffixes=('_f', '_m'))
    return df


def combine_to_create_final(displayer: Displayer, number_min: int = 1000) -> pd.DataFrame:
    # noinspection PyProtectedMember
    raw: pd.DataFrame = displayer._raw
    # noinspection PyProtectedMember
    peaks: pd.DataFrame = displayer._peaks
    # noinspection PyProtectedMember
    age_reference: pd.DataFrame = displayer._age_reference

    raw_wo_unk = raw[~raw.name.isin(UnknownName.get())].copy()
    peaks_wo_unk = peaks[~peaks.name.isin(UnknownName.get())].copy()
    age_ref_wo_unk = age_reference[~age_reference.name.isin(UnknownName.get())].copy()

    total_number = raw_wo_unk[raw_wo_unk.year >= Year.DATA_QUALITY_BEST_AFTER].groupby(
        'name', as_index=False).number.sum().rename(columns=dict(number='total_usages'))

    latest_peaks = peaks_wo_unk.drop_duplicates(['name', 'sex'], keep='last').rename(columns=dict(
        year='peak_year', rank_='peak_rank'))
    peak_cols = ['name', 'peak_year', 'peak_rank']
    latest_peaks = latest_peaks.loc[latest_peaks.sex == 'f', peak_cols].merge(
        latest_peaks.loc[latest_peaks.sex == 'm', peak_cols], on='name', how='outer', suffixes=('_f', '_m'))

    age_percentile_ref1 = build_age_percentile_reference(age_ref_wo_unk, .5)
    age_percentile_ref2 = build_age_percentile_reference(age_ref_wo_unk, .8)
    age_percentile_ref = age_percentile_ref1.merge(age_percentile_ref2, on='name', how='outer', suffixes=('50', '80'))

    ratios = build_gender_ratio_after_year(raw_wo_unk)

    df = total_number.merge(latest_peaks, on='name', how='outer').merge(
        age_percentile_ref, on='name', how='outer').merge(ratios, on='name', how='outer')
    df = df[df.total_usages >= number_min].copy()
    return df


def load_final(recreate: bool = False) -> pd.DataFrame:
    output_filepath: str = 'extras_tn/output.csv'
    if recreate:
        displayer = Displayer()
        displayer.build_base()
        df = combine_to_create_final(displayer)
        df.to_csv(output_filepath, index=False)
    else:
        df = pd.read_csv(output_filepath)
    return df


def filter_final(final: pd.DataFrame, **kwargs) -> pd.DataFrame:
    df: pd.DataFrame = final.copy()

    year: int = kwargs.get('year')
    year_band: int = kwargs.get('yearBand', 5)
    use_peak: bool = kwargs.get('usePeak', True)
    age_ballpark: int = kwargs.get('ageBallpark')
    sex: str = kwargs.get('sex')
    gender_category: tuple[str] = kwargs.get('genderCat')
    number_low: int = kwargs.get('numLo')
    number_high: int = kwargs.get('numHi')

    for integer_col in (
            'peak_year_f', 'peak_rank_f', 'peak_year_m', 'peak_rank_m',
            'middle_lo_f50', 'middle_hi_f50', 'middle_lo_m50', 'middle_hi_m50',
            'middle_lo_f80', 'middle_hi_f80', 'middle_lo_m80', 'middle_hi_m80',
    ):
        df[integer_col] = df[integer_col].fillna(0).map(int)

    final_cols = {'name': 'Name', 'total_usages': f'Total Usages {Year.DATA_QUALITY_BEST_AFTER}-{Year.MAX_YEAR}'}
    if year:
        if use_peak:
            if sex:
                df = df[(df[f'peak_year_{sex}'] >= (year - year_band)) & (df[f'peak_year_{sex}'] <= (year + year_band))]
                final_cols.update({f'peak_year_{sex}': 'Peak Year'})
            else:
                df = df[
                    (df['peak_year_f'] >= (year - year_band)) & (df['peak_year_f'] <= (year + year_band)) &
                    (df['peak_year_m'] >= (year - year_band)) & (df['peak_year_m'] <= (year + year_band))
                    ]
                final_cols.update({'peak_year_f': 'F Peak Year', 'peak_year_m': 'M Peak Year'})
        if age_ballpark:
            if sex:
                df = df[(df[f'middle_lo_{sex}{age_ballpark}'] <= year) & (df[f'middle_hi_{sex}{age_ballpark}'] >= year)]
                final_cols.update({
                    f'middle_lo_{sex}{age_ballpark}': 'Age Ballpark Lower',
                    f'middle_hi_{sex}{age_ballpark}': 'Age Ballpark Upper',
                })
            else:
                df = df[
                    (df[f'middle_lo_f{age_ballpark}'] <= year) & (df[f'middle_hi_f{age_ballpark}'] >= year) &
                    (df[f'middle_lo_m{age_ballpark}'] <= year) & (df[f'middle_hi_m{age_ballpark}'] >= year)
                    ]
                final_cols.update({
                    f'middle_lo_f{age_ballpark}': 'F Age Ballpark Lower',
                    f'middle_hi_f{age_ballpark}': 'F Age Ballpark Upper',
                    f'middle_lo_m{age_ballpark}': 'M Age Ballpark Lower',
                    f'middle_hi_m{age_ballpark}': 'M Age Ballpark Upper',
                })
    final_cols.update({'gender': 'Gender Category'})

    if gender_category:
        remaining_cat_from_input = df.gender.str.split(', ').map(set) - set(gender_category)
        df = df[remaining_cat_from_input.map(len) == 0]  # you want names that had a null set

    if number_low:
        df = df[df.total_usages >= number_low]
    if number_high:
        df = df[df.total_usages <= number_high]

    df = df.sort_values('total_usages', ascending=False)

    df.total_usages = df.total_usages.map(lambda x: f'{x:,}')
    df = df[final_cols.keys()].rename(columns=final_cols)
    return df
