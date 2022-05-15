import os
import re

import numpy as np
import pandas as pd

# years as currently available in dataset
_MIN_YEAR = 1880
MAX_YEAR = int(re.search('^yob([0-9]{4}).txt$', os.listdir('data/names/')[-1]).group(1))
_OUTPUT_RECORDS = False


class Loader:
    def __init__(self, *args, **kwargs):
        self._national_data_directory = 'data/names/'
        self._territories_data_directory = 'data/namesbyterritory/'
        self._first_appearance = None

    def load(self):
        self._read_data()
        self._transform()

    def _read_data(self):
        data = []
        for data_directory, is_territory in [
            (self._national_data_directory, False),
            (self._territories_data_directory, True),
        ]:
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
        self._raw_with_actuarial['number_living'] = (
                self._raw_with_actuarial.number * self._raw_with_actuarial.survival_prob)

    def _read_one_file(self, filename: str, is_territory: bool = None) -> pd.DataFrame:
        df = self._read_one_file_territory(filename) if is_territory else self._read_one_file_national(filename)
        return df

    def _read_one_file_national(self, filename: str) -> pd.DataFrame:
        df = pd.read_csv(self._national_data_directory + filename, names=['name', 'sex', 'number'], dtype={
            'name': str, 'sex': str, 'number': int}).assign(year=filename)
        df.year = df.year.apply(lambda x: x.rsplit('.', 1)[0].replace('yob', '')).apply(int)
        return df

    def _read_one_file_territory(self, filename: str) -> pd.DataFrame:
        df = pd.read_csv(self._territories_data_directory + filename, names=[
            'territory', 'sex', 'year', 'name', 'number'], dtype={
            'territory': str, 'name': str, 'sex': str, 'number': int, 'year': int}).drop(columns=['territory'])
        return df

    @staticmethod
    def _load_actuarial(sex: str) -> pd.DataFrame:
        actuarial = pd.read_csv(f'data/actuarial/{sex}.csv', dtype=int)
        actuarial = actuarial[actuarial.year == MAX_YEAR].copy()
        actuarial['birth_year'] = actuarial.year - actuarial.age
        actuarial['survival_prob'] = actuarial.survivors.apply(lambda x: x / 100000)
        actuarial = actuarial.drop(columns=['year', 'survivors']).rename(columns={'birth_year': 'year'})
        return actuarial


class Displayer(Loader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._after = None
        self._before = None

    def name(
            self,
            name: str,
            after: int = None,
            before: int = None,
            show_historic: bool = None,
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
                    'year': int(peak_by_num.year),
                    'number': int(peak_by_num.number),
                },
                'by_pct': {
                    'year': int(peak_by_pct.year),
                    'pct': float(peak_by_pct.pct_year),
                },
            },
            'latest': {
                'year': int(latest.year),
                'number': int(latest.number),
            },
            'first_appearance': int(self._first_appearance[grouped['name']]),
        }

        if show_historic:
            historic = df[['year', 'number', 'number_f', 'number_m', 'ratio_f', 'ratio_m']].copy()
            for s in ('f', 'm'):
                historic[f'ratio_{s}'] = historic[f'ratio_{s}'].apply(lambda x: round(x, 2))
            name_record.update({'historic': list(historic.to_dict('records')) if _OUTPUT_RECORDS else historic})
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
            number_min: int = None,
            number_max: int = None,
            gender: tuple = None,
            after: int = None,
            before: int = None,
            delta_after: int = None,
            delta_pct: float = None,
            delta_fem: float = None,
            delta_masc: float = None,
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

        # filter on numbers
        if number_min:
            df = df[df.number >= number_min]
        if number_max:
            df = df[df.number <= number_max]

        # filter on length
        if length:
            df = df[df.name.apply(len).apply(lambda x: length[0] <= x <= length[1])]

        # filter on ratio
        if gender:
            df = df[(df.ratio_m >= gender[0]) & (df.ratio_m <= gender[1])]

        # apply text filters
        if pattern:
            df = df[df.name.apply(lambda x: re.search(pattern, x, re.I)).apply(bool)]
        if start:
            df = df[df.name_lower.str.startswith(tuple(i.lower() for i in start))]
        if end:
            df = df[df.name_lower.str.endswith(tuple(i.lower() for i in end))]
        if contains:
            df = df[df.name_lower.apply(lambda x: all((i.lower() in x for i in contains)))]
        if contains_any:
            df = df[df.name_lower.apply(lambda x: any((i.lower() in x for i in contains_any)))]
        if order:
            df = df[df.name_lower.apply(lambda x: re.search('.*'.join(order), x, re.I)).apply(bool)]

        # apply text not-filters
        if not_start:
            df = df[~df.name_lower.str.startswith(tuple(i.lower() for i in not_start))]
        if not_end:
            df = df[~df.name_lower.str.endswith(tuple(i.lower() for i in not_end))]
        if not_contains:
            df = df[~df.name_lower.apply(lambda x: any((i.lower() in x for i in not_contains)))]

        if not len(df):
            return []

        df = df.sort_values('number', ascending=False).drop(columns=['name_lower'])
        for col in ('ratio_f', 'ratio_m'):
            df[col] = df[col].apply(lambda x: round(x, 2))
        df['display'] = [_create_display_ratio(*i) for i in df[['name', 'number', 'ratio_f', 'ratio_m']].to_records(
            index=False)]

        df = df.head(30)
        return df.to_dict('records') if _OUTPUT_RECORDS else df

    def predict_age(
            self,
            name: str,
            gender: str = None,
            living: bool = False,
            buckets: int = None,
    ) -> dict:
        df = self._raw_with_actuarial.copy()
        if living:
            # noinspection PyArgumentList
            df = df.drop(columns=['number']).rename(columns={'number_living': 'number'})

        # filter dataframe
        df = df[df['name'].str.lower() == name.lower()].copy()
        if gender:
            df = df[df.sex == gender.upper()].copy()

        if not len(df):
            return {}

        # calculate cumulative probabilities
        number = df.number.sum()
        if number < 25:
            return {}

        prob = df.groupby('age', as_index=False).number.sum()
        prob['pct'] = prob.number.apply(lambda x: x / number)
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
                'min': float(ages.min()),
                'max': float(ages.max()),
            }

        # create output
        output = {'name': name.title()}
        if gender:
            output['sex'] = gender.upper()
        if living:
            output['living'] = True
        if buckets:
            output['buckets'] = buckets
        output['prediction'] = prediction
        return output

    def predict_gender(
            self,
            name: str,
            year: int = None,
            living: bool = False,
    ) -> dict:
        df = self._raw_with_actuarial.copy()
        if living:
            # noinspection PyArgumentList
            df = df.drop(columns=['number']).rename(columns={'number_living': 'number'})

        # filter dataframe
        df = df[df['name'].str.lower() == name.lower()].copy()
        if year:
            birth_years = list(range(year - 2, year + 3))
            df = df[df.year.isin(birth_years)]
        else:
            birth_years = []

        if not len(df):
            return {}

        # create output
        number = df.number.sum()
        if number < 25:
            return {}

        numbers = df.groupby('sex').number.sum()
        output = {'name': name.title()}
        if birth_years:
            output['birth_year_range'] = birth_years
        if living:
            output['living'] = True
        output.update({
            'prediction': 'F' if numbers.get('F', 0) > numbers.get('M', 0) else 'M',
            'confidence': round(max(numbers.get('F', 0) / number, numbers.get('M', 0) / number), 2),
        })
        return output

    @property
    def _years_to_select(self):
        if self._after and self._before:
            years_range = (self._after, self._before + 1)
        elif self._after:
            years_range = (self._after, MAX_YEAR + 1)
        elif self._before:
            years_range = (_MIN_YEAR, self._before + 1)
        else:
            years_range = (_MIN_YEAR, MAX_YEAR + 1)
        return tuple(range(*years_range))


def _calculate_number_delta(df: pd.DataFrame, **delta) -> pd.DataFrame:
    after = delta.get('after')
    pct = delta.get('pct')

    chg = df[df.year == after].merge(df[df.year == MAX_YEAR], on=['name'], suffixes=('_y1', '_y2'))
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
    chg = chg[chg.year == after].merge(chg[chg.year == MAX_YEAR], on=['name'], suffixes=('_y1', '_y2'))
    if fem_ratio > 0:  # trended fem
        chg['delta'] = chg.ratio_f_y2 >= chg.ratio_f_y1 + fem_ratio
    elif fem_ratio < 0:  # trended masc
        chg['delta'] = chg.ratio_m_y2 >= chg.ratio_m_y1 - fem_ratio
    else:  # no meaningful trend - less than 1% diff
        chg['delta'] = (chg.ratio_f_y1 - chg.ratio_f_y2).apply(abs).apply(lambda x: x <= 0.01)
    df = df[df.name.isin(chg[chg.delta].name)].copy()
    return df


def _create_display_ratio(name: str, number: int, ratio_f: float, ratio_m: float) -> str:
    formatted_number = f'n={number:,}'
    if ratio_f == 1 or ratio_m == 1:
        return f'{name}({formatted_number})'
    elif ratio_f == ratio_m:
        return f'{name}({formatted_number}, no lean)'
    elif ratio_f > ratio_m:
        return f'{name}({formatted_number}, f={ratio_f})'
    else:  # m > f
        return f'{name}({formatted_number}, m={ratio_m})'
