import os
import re

import numpy as np
import pandas as pd

# years as currently available in dataset
_MIN_YEAR = 1880
_MAX_YEAR = int(re.search('^yob([0-9]{4}).txt$', os.listdir('data/names/')[-1]).group(1))
_OUTPUT_RECORDS = False


class Loader:
    def __init__(self, **kwargs):
        self._national_data_directory = 'data/names/'
        self._territories_data_directory = 'data/namesbyterritory/'
        self._first_appearance = None

    def load(self):
        self._read_data()
        self._transform()

    def _read_data(self):
        directories = {
            self._national_data_directory: False,
            self._territories_data_directory: True,
        }
        data = []
        for data_directory, is_territory in directories.items():
            for filename in os.listdir(data_directory):
                if not filename.lower().endswith('.txt'):
                    continue
                data.append(self._read_one_file(filename, is_territory))
        self._concatenated = pd.concat(data)

    def _transform(self):
        # combine territories w/ national
        self._raw = self._concatenated.groupby(['name', 'sex', 'year'], as_index=False).number.sum()
        self._number_per_year = self._concatenated.groupby('year', as_index=False).number.sum()

        # name by year
        self._name_by_year = self._concatenated.groupby(['name', 'year'], as_index=False).number.sum().merge(
            self._number_per_year, on=['year'], suffixes=('', '_total'))
        self._name_by_year['pct_year'] = self._name_by_year.number / self._name_by_year.number_total
        self._name_by_year = self._name_by_year.drop(columns=['number_total'])

        # first appearance
        self._first_appearance = self._raw.groupby('name').year.min()

        # add ratios
        _separate = lambda x: self._raw[self._raw.sex == x].drop(columns=['sex'])
        self._calcd = _separate('F').merge(_separate('M'), on=['name', 'year'], suffixes=(
            '_f', '_m'), how='outer').merge(self._name_by_year, on=['name', 'year'])
        for s in ('f', 'm'):
            self._calcd[f'number_{s}'] = self._calcd[f'number_{s}'].fillna(0).apply(int)
            self._calcd[f'ratio_{s}'] = self._calcd[f'number_{s}'] / self._calcd['number']

        # add actuarial - loses years before 1900
        self._raw_with_actuarial = pd.concat(self._raw[self._raw.sex == s.upper()].merge(self._load_actuarial(s), on=[
            'year']) for s in ('f', 'm'))
        self._raw_with_actuarial.number = self._raw_with_actuarial.number * self._raw_with_actuarial.survival_prob

    def _read_one_file(self, filename, is_territory=None):
        df = self._read_one_file_territory(filename) if is_territory else self._read_one_file_national(filename)
        return df

    def _read_one_file_national(self, filename):
        df = pd.read_csv(self._national_data_directory + filename, names=['name', 'sex', 'number'], dtype={
            'name': str, 'sex': str, 'number': int}).assign(year=filename)
        df.year = df.year.apply(lambda x: x.rsplit('.', 1)[0].replace('yob', '')).apply(int)
        return df

    def _read_one_file_territory(self, filename):
        df = pd.read_csv(self._territories_data_directory + filename, names=[
            'territory', 'sex', 'year', 'name', 'number'], dtype={
            'territory': str, 'name': str, 'sex': str, 'number': int, 'year': int}).drop(columns=['territory'])
        return df

    @staticmethod
    def _load_actuarial(sex: str):
        actuarial = pd.read_csv(f'data/actuarial/{sex}.csv', dtype=int)
        actuarial = actuarial[actuarial.year == _MAX_YEAR].copy()
        actuarial['birth_year'] = actuarial.year - actuarial.age
        actuarial['survival_prob'] = actuarial.survivors.apply(lambda x: x / 100000)
        actuarial = actuarial.drop(columns=['year', 'survivors']).rename(columns={'birth_year': 'year'})
        return actuarial


class Displayer(Loader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._after = kwargs.get('after')  # after this year (inclusive)
        self._before = kwargs.get('before')  # before this year (inclusive)

    def info(
            self,
            name: str,
            after: int = None,
            before: int = None,
    ) -> dict:
        # set up
        self._after = after
        self._before = before
        df = self._calcd.copy()

        # filter on name
        df = df[df['name'].str.lower() == name.lower()]
        if not len(df):
            return {}

        # create metadata dfs
        peak_by_num = df.loc[df.number.idxmax()].copy()
        peak_by_pct = df.loc[df.pct_year.idxmax()].copy()
        latest = df.loc[df.year.idxmax()].copy()

        # filter on years
        df = df[df.year.isin(self._years_to_select)]
        if not len(df):
            return {}

        df = df.sort_values('year')

        # aggregate
        grouped = df.groupby('name', as_index=False).agg({'number': sum, 'number_f': sum, 'number_m': sum})
        for s in ('f', 'm'):
            grouped[f'ratio_{s}'] = (grouped[f'number_{s}'] / grouped.number).apply(lambda x: round(x, 2))

        # do final computations
        historic_numbers = df[['year', 'number', 'number_f', 'number_m']].copy()
        historic_ratios = df[['year', 'ratio_f', 'ratio_m']].copy()
        for s in ('f', 'm'):
            historic_ratios[f'ratio_{s}'] = historic_ratios[f'ratio_{s}'].apply(lambda x: round(x, 2))

        # create output
        grouped = grouped.to_dict('records')[0]
        name_record = {
            'name': grouped['name'],
            'numbers': {
                'total': grouped['number'],
                'f': grouped['number_f'],
                'm': grouped['number_m'],
            },
            'ratios': {
                'f': grouped['ratio_f'],
                'm': grouped['ratio_m'],
            },
            'peak': {
                'by_number': {
                    'year': peak_by_num.year,
                    'number': peak_by_num.number,
                },
                'by_pct': {
                    'year': peak_by_pct.year,
                    'pct': peak_by_pct.pct_year,
                },
            },
            'latest': {
                'year': latest.year,
                'number': latest.number,
            },
            'first_appearance': {
                'year': self._first_appearance[grouped['name']],
            },
            'historic': {
                'number': list(historic_numbers.to_dict('records')) if _OUTPUT_RECORDS else historic_numbers,
                'ratio': list(historic_ratios.to_dict('records')) if _OUTPUT_RECORDS else historic_ratios,
            },
        }
        return name_record

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
            length: tuple = None,
            fem: (bool, tuple) = None,
            masc: (bool, tuple) = None,
            neu: (bool, tuple) = None,
            delta_after: int = None,
            delta_pct: float = None,
            delta_fem: float = None,
            delta_masc: float = None,
            number: tuple = None,
            after: int = None,
            before: int = None,
    ) -> list:
        # set up
        self._after = after
        self._before = before
        df = self._calcd.copy()

        # calculate number/gender delta
        if delta_after:
            if delta_pct is not None:
                df = _calculate_number_delta(df, after=delta_after, pct=delta_pct)
            if delta_fem is None and delta_masc is not None:
                delta_fem = -delta_masc
            if delta_fem is not None:
                df = _calculate_gender_delta(df, after=delta_after, fem_ratio=delta_fem)

        # filter on years
        df = df[df.year.isin(self._years_to_select)].copy()

        # aggregate
        df = df.groupby('name', as_index=False).agg({'number': sum, 'number_f': sum, 'number_m': sum})
        for s in ('f', 'm'):
            df[f'ratio_{s}'] = df[f'number_{s}'] / df.number

        # add lowercase name for filtering
        df['name_lower'] = df.name.apply(lambda x: x.lower())

        # filter on number
        if number:
            df = df[(df.number >= number[0]) & (df.number <= number[1])]

        # filter on length
        if length:
            df = df[df.name.apply(len).isin(length)]

        # set fem/masc lean filters
        if fem is True:
            fem = (0.5, 1)
        elif masc is True:
            masc = (0.5, 1)
        elif neu is True:
            fem = (0.25, 0.75)
        elif neu is not None:
            fem = (0.5 - neu, 0.5 + neu)

        # filter on ratio
        if fem is not None:
            df = df[(df.ratio_f >= fem[0]) & (df.ratio_f <= fem[1])]
        elif masc is not None:
            df = df[(df.ratio_m >= masc[0]) & (df.ratio_m <= masc[1])]

        # apply text filters
        if pattern is not None:
            df = df[df.name.apply(lambda x: re.search(pattern, x, re.I)).apply(bool)]
        if start is not None:
            df = df[df.name.apply(lambda x: re.search('^({})'.format('|'.join(start)), x, re.I)).apply(bool)]
        if end is not None:
            df = df[df.name.apply(lambda x: re.search('({})$'.format('|'.join(end)), x, re.I)).apply(bool)]
        if contains is not None:
            df = df[df.name_lower.apply(lambda x: all((i.lower() in x for i in contains)))]
        if contains_any is not None:
            df = df[df.name.apply(lambda x: re.search('|'.join(contains_any), x, re.I)).apply(bool)]
        if order is not None:
            df = df[df.name_lower.apply(lambda x: re.search('.*'.join(order), x)).apply(bool)]

        # apply text not-filters
        _normalize_type_or = lambda x: tuple(char.lower() for char in x)
        if not_start is not None:
            df = df[~df.name_lower.str.startswith(_normalize_type_or(not_start))]
        if not_end is not None:
            df = df[~df.name.str.endswith(_normalize_type_or(not_end))]
        if not_contains is not None:
            df = df[~df.name_lower.str.contains('|'.join(not_contains).lower())]

        if not len(df):
            return []

        summary = df.sort_values('number', ascending=False).drop(columns=['name_lower'])
        return summary.to_dict('records') if _OUTPUT_RECORDS else summary

    def predict_age(
            self,
            name: str,
            sex: str = None,
            exclude_deceased: bool = False,
            buckets: int = None,
    ) -> dict:
        df = self._raw_with_actuarial.copy() if exclude_deceased else self._raw.copy()

        # filter dataframe
        df = df[df['name'].str.lower() == name.lower()].copy()
        if sex:
            df = df[df.sex == sex.upper()].copy()

        if not len(df):
            return {}

        # calculate cumulative probabilities
        total = df.number.sum()
        prob = df.groupby('age', as_index=False).number.sum()
        prob['pct'] = prob.number.apply(lambda x: x / total)
        prob = prob.sort_values('pct', ascending=False)
        prob['cumulative'] = prob.pct.cumsum()
        prediction = {
            'mean': int(round(df.groupby(df.name).apply(lambda x: np.average(x.age, weights=x.number)).values[0])),
            'percentiles': {},
        }
        percentiles = tuple(round(i / buckets, 2) for i in range(1, buckets + 1)) if buckets else (0.68, 0.95, 0.997)
        for percentile in percentiles:
            ages = prob[prob.cumulative <= percentile].age
            if not len(ages):
                continue
            prediction['percentiles'][percentile] = {
                'min': ages.min(),
                'max': ages.max(),
            }

        # create output
        output = {
            'name': name.title(),
            'sex': sex.upper() if sex else None,
            'exclude_deceased': bool(exclude_deceased),
            'buckets': buckets,
            'prediction': prediction,
        }
        return output

    def predict_gender(
            self,
            name: str,
            birth_year: int = None,
            exclude_deceased: bool = False,
    ) -> dict:
        df = self._raw_with_actuarial.copy() if exclude_deceased else self._raw.copy()

        # filter dataframe
        df = df[df['name'].str.lower() == name.lower()].copy()
        if birth_year:
            birth_years = list(range(birth_year - 2, birth_year + 3))
            df = df[df.year.isin(birth_years)]
        else:
            birth_years = []

        if not len(df):
            return {}

        # create output
        number = df.number.sum()
        numbers = df.groupby('sex').number.sum()
        output = {
            'name': name.title(),
            'birth_year_range': birth_years,
            'exclude_deceased': bool(exclude_deceased),
            'prediction': 'F' if numbers['F'] > numbers['M'] else 'M',
            'confidence': round(max(numbers['F'] / number, numbers['M'] / number), 2),
        }
        return output

    @property
    def _years_to_select(self):
        if self._after and self._before:
            years_range = (self._after, self._before + 1)
        elif self._after:
            years_range = (self._after, _MAX_YEAR + 1)
        elif self._before:
            years_range = (_MIN_YEAR, self._before + 1)
        else:
            years_range = (_MIN_YEAR, _MAX_YEAR + 1)
        return tuple(range(*years_range))


def _calculate_number_delta(df: pd.DataFrame, **delta) -> pd.DataFrame:
    after = delta.get('after')
    pct = delta.get('pct')

    chg = df[df.year == after].merge(df[df.year == _MAX_YEAR], on=['name'], suffixes=('_y1', '_y2'))
    if pct > 0:  # trended up
        chg['delta'] = chg.pct_year_y2 >= chg.pct_year_y1 * (1 + pct)
    elif pct < 0:  # trended down
        chg['delta'] = chg.pct_year_y1 >= chg.pct_year_y2 * (1 - pct)
    else:  # no meaningful trend - less than 1% diff
        chg['delta'] = (chg.pct_year_y1 / chg.pct_year_y2).apply(lambda x: 0.99 <= x <= 1.01)
    df = df[df.name.isin(chg[chg.delta].name)].copy()
    return df


def _calculate_gender_delta(df: pd.DataFrame, **delta) -> pd.DataFrame:
    after = delta.get('after')
    fem_ratio = delta.get('fem_ratio')

    chg = df.copy()
    chg = chg[chg.year == after].merge(chg[chg.year == _MAX_YEAR], on=['name'], suffixes=('_y1', '_y2'))
    if fem_ratio > 0:  # trended fem
        chg['delta'] = chg.ratio_f_y2 >= chg.ratio_f_y1 + fem_ratio
    elif fem_ratio < 0:  # trended masc
        chg['delta'] = chg.ratio_m_y2 >= chg.ratio_m_y1 - fem_ratio
    else:  # no meaningful trend - less than 1% diff
        chg['delta'] = (chg.ratio_f_y1 - chg.ratio_f_y2).apply(abs).apply(lambda x: x <= 0.01)
    df = df[df.name.isin(chg[chg.delta].name)].copy()
    return df
