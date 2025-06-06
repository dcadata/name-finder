import os
import re
import string
from enum import Enum

import pandas as pd
import seaborn as sns

from demos import SsaSex


class Filepath:
    DATA_DIR: str = 'data/'
    NATIONAL_DATA_DIR: str = 'data/names/'
    TERRITORIES_DATA_DIR: str = 'data/namesbyterritory/'
    ACTUARIAL: str = 'data/actuarial/{sex}.csv'
    APPLICANTS_DATA: str = 'data/applicants/data.csv'
    AGE_PREDICTION_REFERENCE: str = 'data/generated/age_prediction_reference.csv'
    GENDER_PREDICTION_REFERENCE: str = 'data/generated/gender_prediction_reference.csv'
    TOTAL_NUMBER_LIVING_REFERENCE: str = 'data/generated/raw_with_actuarial.total_number_living.csv'


class Pattern:
    YEAR: str = '^yob([0-9]{4}).txt$'


class Year:
    MIN_YEAR: int = 1880
    MAX_YEAR: int = int(re.search(Pattern.YEAR, os.listdir(Filepath.NATIONAL_DATA_DIR)[-1]).group(1))
    DATA_QUALITY_BEST_AFTER: int = 1937

    @classmethod
    def get_default_years_as_dict(cls) -> dict[str, int]:
        return dict(after=Year.DATA_QUALITY_BEST_AFTER, before=Year.MAX_YEAR)


class UnknownName(Enum):
    Unknown = 'Unknown'
    Infant = 'Infant'
    Baby = 'Baby'
    Unnamed = 'Unnamed'
    Unborn = 'Unborn'
    Notnamed = 'Notnamed'
    Newborn = 'Newborn'

    @classmethod
    def get(cls) -> tuple:
        return tuple(i.value for i in cls)


class DFAgg:
    NUMBER_SUM = dict(number='sum', number_f='sum', number_m='sum')


class Builder:
    def __init__(self) -> None:
        self._raw: pd.DataFrame
        self._age_reference: pd.DataFrame
        self._applicants_data: pd.DataFrame
        self._name_by_year: pd.DataFrame
        self._peaks: pd.DataFrame
        self._calcd: pd.DataFrame
        self.raw_with_actuarial: pd.DataFrame

    def build_base(self) -> None:
        self._load_name_data()
        self._load_applicants_data()
        self._build_name_by_year()
        self._build_peaks()
        self._build_calcd_with_ratios_and_number_pct()
        self._build_raw_with_actuarial()
        self._load_predict_age_reference()
        return

    def _load_name_data(self) -> None:
        self._raw = pd.concat([
            _load_name_data_for_one_year(filename) for filename in os.listdir(Filepath.NATIONAL_DATA_DIR)
            if filename.lower().endswith('.txt')
        ])
        self._raw.sex = self._raw.sex.str.lower()
        self._raw.rank_ = self._raw.rank_.map(int)
        return

    def _load_predict_age_reference(self) -> None:
        dtype = dict(name=str, sex=str, year=int, number_living_pct=float)
        self._age_reference = pd.read_csv(Filepath.AGE_PREDICTION_REFERENCE, usecols=list(dtype.keys()), dtype=dtype)
        return

    def _load_applicants_data(self) -> None:
        self._applicants_data = pd.read_csv(Filepath.APPLICANTS_DATA, dtype=int)
        return

    def _build_name_by_year(self) -> None:
        self._name_by_year = self._raw.groupby(['name', 'year'], as_index=False).number.sum()
        self._name_by_year['rank_'] = self._name_by_year.groupby('year').number.rank(method='min', ascending=False)
        return

    def _build_peaks(self) -> None:
        peaks_base = pd.concat((self._raw, self._name_by_year.assign(sex=SsaSex.All)))
        peaks_base = peaks_base[peaks_base.year >= Year.DATA_QUALITY_BEST_AFTER]
        self._peaks = peaks_base.groupby(['name', 'sex'], as_index=False).agg(dict(rank_='min')).merge(
            peaks_base, on=['name', 'sex', 'rank_'], how='left').sort_values('year')
        self._peaks.rank_ = self._peaks.rank_.map(int)
        return

    def _build_calcd_with_ratios_and_number_pct(self) -> None:
        separate_by_sex = lambda x: self._raw[self._raw.sex == x].drop(columns='sex').rename(columns=dict(rank_='rank'))
        merge_on = ['name', 'year']
        self._calcd = (
            separate_by_sex(SsaSex.Female)
            .merge(separate_by_sex(SsaSex.Male), on=merge_on, suffixes=SsaSex.Suffix, how='outer')
            .merge(self._name_by_year, on=merge_on).sort_values('year')
        )
        for s in SsaSex.Both:
            self._calcd[f'number_{s}'] = self._calcd[f'number_{s}'].fillna(0).map(int)
            self._calcd[f'rank_{s}'] = self._calcd[f'rank_{s}'].fillna(-1).map(int)
        self._calcd.rank_ = self._calcd.rank_.map(int)

        self._calcd = self._calcd.merge(self._applicants_data, on='year', suffixes=('', '_total'))
        for s in SsaSex.Both:
            self._calcd[f'number_pct_{s}'] = self._calcd[f'number_{s}'] / self._calcd[f'number_{s}_total']
        self._calcd['number_pct'] = self._calcd.number / self._calcd.number_total
        self._calcd = self._calcd.drop(columns=['number_f_total', 'number_m_total', 'number_total'])
        return

    def _build_raw_with_actuarial(self) -> None:
        # loses years before 1900
        self.raw_with_actuarial = self._raw.merge(_load_actuarial_data(), on=['sex', 'year'])
        self.raw_with_actuarial[
            'number_living'] = self.raw_with_actuarial.number * self.raw_with_actuarial.survival_prob
        return

    @property
    def calculated(self) -> pd.DataFrame:
        return self._calcd


class Displayer(Builder):
    def name(
            self,
            name: str,
            after: int = None,
            before: int = None,
            year: int = None,
            display: bool | str = None,
    ) -> dict:
        df = self._calcd.copy()

        # filter on name
        name = _standardize_name(name)
        df = df[df.name == name]
        if not len(df):
            return {}

        # build metadata
        selected_year = {'selected_year': _restructure_earliest_or_latest(df[df.year == year].iloc[0])} if year else {}
        earliest, latest = df.iloc[[0, -1]].to_dict('records')

        # filter on years
        df = _filter_on_years(df, year, after, before)
        if not len(df):
            return {}

        if display:
            _make_plot_for_name(df, name, display)

        # aggregate
        grouped = df.groupby('name', as_index=False).agg(DFAgg.NUMBER_SUM)
        for s in SsaSex.Both:
            grouped[f'ratio_{s}'] = (grouped[f'number_{s}'] / grouped.number).round(3)

        # build output
        grouped = grouped.iloc[0].to_dict()
        output = {
            'name': grouped['name'],
            **dict(after=after, before=before, year=year),
            'numbers': {
                SsaSex.Total: grouped['number'],
                SsaSex.Female: grouped['number_f'],
                SsaSex.Male: grouped['number_m'],
            },
            'ratios': {
                SsaSex.Female: grouped['ratio_f'],
                SsaSex.Male: grouped['ratio_m'],
            },
            'peak': self.get_peaks(name),
            'latest': _restructure_earliest_or_latest(latest),
            'earliest': _restructure_earliest_or_latest(earliest),
            **selected_year,
        }
        return output

    def search(
            self,
            pattern: str = None,
            start: tuple = None,
            end: tuple = None,
            contains: tuple = None,
            contains_any: tuple = None,
            not_start: tuple = None,
            not_end: tuple = None,
            not_contains: tuple = None,
            order: tuple = None,
            length_min: int = None,
            length_max: int = None,
            number_min: int = None,
            number_max: int = None,
            gender: tuple[float, float] = None,
            after: int = None,
            before: int = None,
            year: int = None,
            peaked: pd.DataFrame = None,
            top: int = 20,
            sort_sex: str = None,
            display: bool = False,
    ) -> pd.DataFrame | list:
        df = self._calcd.copy()
        # exclude placeholder names
        df = df[~df.name.isin(UnknownName.get())].copy()

        # filter on years
        df = _filter_on_years(df, year, after, before).copy()

        # aggregate
        agg_fields = DFAgg.NUMBER_SUM.copy()
        if year:
            agg_fields.update(dict(rank_='min', rank_f='min', rank_m='min'))
        df = df.groupby('name', as_index=False).agg(agg_fields)
        for s in SsaSex.Both:
            df[f'ratio_{s}'] = df[f'number_{s}'] / df.number

        # add lowercase name for filtering
        df['name_lower'] = df.name.str.lower()

        # filter on numbers
        if number_min:
            df = df[df.number >= number_min]
        if number_max:
            df = df[df.number <= number_max]

        # filter on length
        if length_min or length_max:
            if length_min:
                df = df[df.name.map(len) >= length_min]
            if length_max:
                df = df[df.name.map(len) <= length_max]

        # filter on ratio
        if gender:
            df = df[(df.ratio_m >= gender[0]) & (df.ratio_m <= gender[1])]

        # apply text filters
        if pattern:
            df = df[df.name.apply(lambda x: re.search(pattern, x, re.I)).map(bool)]
        if start:
            df = df[df.name_lower.str.startswith(tuple(i.lower() for i in start))]
        if end:
            df = df[df.name_lower.str.endswith(tuple(i.lower() for i in end))]
        if contains:
            df = df[df.name_lower.apply(lambda x: all((i.lower() in x for i in contains)))]
        if contains_any:
            df = df[df.name_lower.apply(lambda x: any((i.lower() in x for i in contains_any)))]
        if order:
            df = df[df.name_lower.apply(lambda x: re.search('.*'.join(order), x, re.I)).map(bool)]

        # apply text not-filters
        if not_start:
            df = df[~df.name_lower.str.startswith(tuple(i.lower() for i in not_start))]
        if not_end:
            df = df[~df.name_lower.str.endswith(tuple(i.lower() for i in not_end))]
        if not_contains:
            df = df[~df.name_lower.apply(lambda x: any((i.lower() in x for i in not_contains)))]

        if not len(df):
            return df

        sort_field = f'number_{sort_sex}' if sort_sex else 'number'
        df = df.sort_values(sort_field, ascending=False).drop(columns='name_lower')

        if peaked is not None:
            df = df[df.name.isin(peaked.name)].copy()

        if top:
            df = df.head(top).copy()

        if display:
            return [_make_search_display_string(*i) for i in df[['name', 'number', 'ratio_f', 'ratio_m']].to_records(
                index=False)]
        return df

    def predict_age(self, name: str, sex: str, mid_percentile: float = .68) -> pd.DataFrame:
        name = _standardize_name(name)
        lower_percentile = .5 - mid_percentile / 2
        upper_percentile = 1 - lower_percentile

        df = self._age_reference.copy()
        df = df[(df.name == name) & (df.sex == sex)].drop(columns='sex')

        df.number_living_pct = df.number_living_pct.cumsum()
        df['lower'] = (lower_percentile - df.number_living_pct).abs()
        df['upper'] = (upper_percentile - df.number_living_pct).abs()

        df = (
            df[(df.lower == df.lower.min()) | (df.upper == df.upper.min())]
            .sort_values('year')
            .assign(bound=['lower', 'upper'])
            .assign(percentile=[lower_percentile, upper_percentile])
            .set_index('bound')
        )[['percentile', 'year']]
        percentile_band = df.percentile.upper - df.percentile.lower
        year_band = df.year.upper - df.year.lower
        df = df.T.assign(band=[percentile_band, year_band]).T
        df.year = df.year.map(int)
        return df

    def predict_gender(
            self,
            name: str,
            after: int = None,
            before: int = None,
            year: int = None,
            living: bool = True,
    ) -> dict:
        # set up
        name = _standardize_name(name)
        output: dict[str, str | bool | int | float] = dict(name=name)
        df = self.raw_with_actuarial.copy()

        if living:
            df = df.drop(columns='number').rename(columns={'number_living': 'number'})
            output['living'] = True

        # filter dataframe
        df = df[df.name == name].copy()
        if year:
            df = df[df.year == year]
            output['year'] = year
        else:
            if after:
                df = df[df.year >= after]
                output['after'] = after
            if before:
                df = df[df.year <= before]
                output['before'] = before

        # add to output
        number = df.number.sum()
        output['number'] = int(number)

        if number:
            numbers = df.groupby('sex').number.sum()
            prediction = SsaSex.Female if numbers.get(SsaSex.Female, 0) > numbers.get(SsaSex.Male, 0) else SsaSex.Male
            output.update(dict(
                prediction=prediction,
                confidence=round(numbers[prediction] / number, 2),
            ))

        return output

    def filter_peaks_or_raw(self, **kwargs) -> pd.DataFrame:
        use_raw: bool = kwargs.get('use_raw')
        df = (self._raw if use_raw else self._peaks).copy()
        if after := kwargs.get('after'):
            df = df[df.year >= after]
        if before := kwargs.get('before'):
            df = df[df.year <= before]
        if year := kwargs.get('year'):
            df = df[df.year == year]
        if sex := kwargs.get('sex', None if use_raw else SsaSex.All):
            df = df[df.sex == sex]
        if rank_min := kwargs.get('rank_min'):
            df = df[df.rank_ >= rank_min]
        if rank_max := kwargs.get('rank_max'):
            df = df[df.rank_ <= rank_max]
        return df

    def get_peaks(self, name: str) -> pd.DataFrame:
        return self._peaks[self._peaks.name == name].groupby(['sex', 'year'], as_index=False).agg(dict(
            rank_='min', number='max')).sort_values(['sex', 'year']).to_dict('records')


def _load_name_data_for_one_year(filename: str) -> pd.DataFrame:
    year = re.search(Pattern.YEAR, filename).group(1)
    dtypes = dict(name=str, sex=str, number=int)
    df = pd.read_csv(Filepath.NATIONAL_DATA_DIR + filename, names=list(dtypes.keys()), dtype=dtypes).assign(year=year)
    df.year = df.year.map(int)
    df['rank_'] = df.groupby('sex').number.rank(method='min', ascending=False)
    return df


def _load_actuarial_data() -> pd.DataFrame:
    actuarial = pd.concat(pd.read_csv(Filepath.ACTUARIAL.format(sex=s), usecols=[
        'year', 'age', 'survivors'], dtype=int).assign(sex=s) for s in SsaSex.Both)
    actuarial = actuarial[actuarial.year == Year.MAX_YEAR].copy()
    actuarial['birth_year'] = actuarial.year - actuarial.age
    actuarial['survival_prob'] = actuarial.survivors / 100_000
    actuarial = actuarial.drop(columns=['year', 'survivors']).rename(columns={'birth_year': 'year'})
    return actuarial


def _make_plot_for_name(df: pd.DataFrame, name: str, display: bool | str) -> None:
    value_field_name = 'number' if type(display) == bool else display
    year_field = 'year'
    display_fields = list(map(lambda x: f'{value_field_name}_{x}', SsaSex.Both))
    historic = df[[year_field, *display_fields]].melt([year_field], display_fields, '', value_field_name)
    historic[''] = historic[''].str.slice(-1)
    ax = sns.lineplot(historic, x='year', y=value_field_name, hue='', **SsaSex.HueOrderAndPalette)
    ax.set_title(name)
    ax.figure.tight_layout()
    return


def _filter_on_years(df: pd.DataFrame, year: int = None, after: int = None, before: int = None) -> pd.DataFrame:
    if year:
        df = df[df.year == year]
        return df
    if after:
        df = df[df.year >= after]
    if before:
        df = df[df.year <= before]
    return df


def _make_display_ratio(ratio_f: float, ratio_m: float, ignore_ones: bool = False) -> str:
    if ignore_ones and (ratio_f == 1 or ratio_m == 1):
        return ''
    elif ratio_f > ratio_m:
        return f'f={int(round(ratio_f * 100))}%'
    elif ratio_m > ratio_f:
        return f'm={int(round(ratio_m * 100))}%'
    else:  # they're equal
        return 'no lean'


def _make_search_display_string(name: str, number: int, ratio_f: float, ratio_m: float) -> str:
    if display_ratio := _make_display_ratio(ratio_f, ratio_m):
        display_ratio = '; ' + display_ratio
    return f'{name} (n={number:,}{display_ratio})'


def _restructure_earliest_or_latest(earliest: dict) -> dict:
    return dict(
        year=earliest['year'],
        number=dict(total=earliest['number'], f=earliest['number_f'], m=earliest['number_m']),
        rank=dict(f=earliest['rank_f'], m=earliest['rank_m']),
    )


def _standardize_name(name: str) -> str:
    reference = {
        'a': 'à|á',
        'c': 'ç',
        'e': 'è|é|ê|ë',
        'i': 'í|î',
        'n': 'ñ',
        'o': 'ó|ô',
        'u': 'ù|ú|ü',
    }
    name = name.lower()
    for deacc, acc in reference.items():
        name = re.sub(acc, deacc, name)
    return ''.join(re.findall(f'[{string.ascii_lowercase}]+', name)).title()


def build_predict_gender_reference(
        displayer: Displayer,
        after: int = None,
        before: int = None,
        ratio_min: float = .8,
) -> pd.DataFrame:
    df = displayer.calculated.copy()

    if after:
        df = df[df.year >= after]
    if before:
        df = df[df.year <= before]

    df = df.groupby('name', as_index=False).agg(DFAgg.NUMBER_SUM)

    df.loc[df.number_f > df.number_m, 'gender_prediction'] = 'f'
    df.loc[df.number_f < df.number_m, 'gender_prediction'] = 'm'
    df.loc[df.number_f == df.number_m, 'gender_prediction'] = 'x'

    if ratio_min:
        ratio_f = df.number_f / df.number
        ratio_m = df.number_m / df.number
        df.loc[(ratio_f < ratio_min) & (ratio_m < ratio_min), 'gender_prediction'] = 'x'

    df.loc[df.number < 25, 'gender_prediction'] = 'rare'
    df.gender_prediction = df.gender_prediction.fillna('unk')

    df = df[['name', 'gender_prediction']]
    df.to_csv(Filepath.GENDER_PREDICTION_REFERENCE, index=False)
    return df


def build_total_number_living_from_actuarial(raw_with_actuarial: pd.DataFrame) -> None:
    total_number_living = raw_with_actuarial.groupby(['name', 'sex'], as_index=False).number_living.sum()
    total_number_living.to_csv(Filepath.TOTAL_NUMBER_LIVING_REFERENCE, index=False)
    return


def _read_total_number_living() -> pd.DataFrame:
    dtype = dict(name=str, sex=str, number_living=float)
    return pd.read_csv(Filepath.TOTAL_NUMBER_LIVING_REFERENCE, usecols=list(dtype.keys()), dtype=dtype)


def build_predict_age_reference(raw_with_actuarial: pd.DataFrame) -> None:
    ref = raw_with_actuarial[['name', 'sex', 'year', 'number_living']].copy()
    ref = ref.groupby(['name', 'sex', 'year'], as_index=False).number_living.sum().merge(
        _read_total_number_living(), on=['name', 'sex'], suffixes=('', '_name'))
    ref = ref[ref.number_living_name >= 20].copy()
    ref['number_living_pct'] = ref.number_living / ref.number_living_name
    ref = ref.drop(columns=['number_living', 'number_living_name']).sort_values('year')
    ref.to_csv(Filepath.AGE_PREDICTION_REFERENCE, index=False)
    return


def build_all_generated_data() -> None:
    displayer = Displayer()
    displayer.build_base()

    build_predict_gender_reference(displayer)
    build_total_number_living_from_actuarial(displayer.raw_with_actuarial)
    build_predict_age_reference(displayer.raw_with_actuarial)
    return
