import os
import re

import pandas as pd

# years as currently available in dataset
_MIN_YEAR = 1880
_MAX_YEAR = int(re.search('^yob([0-9]{4}).txt$', os.listdir('data/names/')[-1]).group(1))


class Calculator:
    def __init__(self, **kwargs):
        self._national_data_directory = 'data/names/'
        self._territories_data_directory = 'data/namesbyterritory/'

    def calculate(self):
        self._read_data()
        self._organize_data()
        self._add_ratios()
        del self._raw_dataframes

    def _read_data(self):
        data_to_read = {
            self._national_data_directory: False,
            self._territories_data_directory: True,
        }
        self._raw_dataframes = []
        for data_directory, is_territory in data_to_read.items():
            for filename in os.listdir(data_directory):
                if not filename.lower().endswith('.txt'):
                    continue
                self._raw_dataframes.append(self._read_one_file(filename, is_territory))

    def _organize_data(self):
        concatenated = pd.concat(self._raw_dataframes)
        # combine territories w/ national
        self._raw = concatenated.groupby(['name', 'sex', 'year'], as_index=False).number.sum()
        self._number_per_year = concatenated.groupby('year', as_index=False).number.sum()

        # name by year
        self._name_by_year = concatenated.groupby(['name', 'year'], as_index=False).number.sum().merge(
            self._number_per_year, on=['year'], suffixes=('', '_total'))
        self._name_by_year['pct_year'] = self._name_by_year.number / self._name_by_year.number_total
        self._name_by_year = self._name_by_year.drop(columns=['number_total'])

        # name by sex by year
        self._name_by_sex_by_year = self._raw.merge(self._number_per_year, on=['year'], suffixes=('', '_total'))
        self._name_by_sex_by_year['pct_year'] = (
                self._name_by_sex_by_year.number / self._name_by_sex_by_year.number_total)
        self._name_by_sex_by_year = self._name_by_sex_by_year.drop(columns=['number_total'])

        # first appearance
        self._first_appearance = self._raw.groupby('name', as_index=False).year.min()

    def _add_ratios(self):
        self.calcd = self._raw.copy()
        _separate = lambda x: self.calcd[self.calcd.sex == x].drop(columns=['sex'])
        self.calcd = _separate('F').merge(_separate('M'), on=['name', 'year'], suffixes=(
            '_f', '_m'), how='outer').merge(self._name_by_year, on=['name', 'year'])
        for s in ('f', 'm'):
            self.calcd[f'number_{s}'] = self.calcd[f'number_{s}'].fillna(0).apply(int)
            self.calcd[f'ratio_{s}'] = self.calcd[f'number_{s}'] / self.calcd['number']

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


class Displayer(Calculator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._after = kwargs.get('after')  # after this year (inclusive)
        self._before = kwargs.get('before')  # before this year (inclusive)
        self._top = kwargs.get('top', 25)  # if searching, number of results to display

    def add_name(
            self,
            name: str,
            after: int = None,
            before: int = None,
    ):
        # set up
        self._after = after
        self._before = before
        df = self.calcd.copy()

        # filter on name
        df = df[df['name'].str.lower() == name.lower()]
        if not len(df):
            return ''

        # create metadata dfs
        peak_by_num = df.loc[df.number.idxmax()].copy()
        peak_by_pct = df.loc[df.pct_year.idxmax()].copy()
        latest = df.loc[df.year.idxmax()].copy()

        # filter on years
        df = df[df.year.isin(self._years_to_select)]
        if not len(df):
            return ''

        df = df.sort_values('year')

        # aggregate
        name_record = df.groupby('name', as_index=False).agg({'number': sum, 'number_f': sum, 'number_m': sum})
        for s in ('f', 'm'):
            name_record[f'ratio_{s}'] = name_record[f'number_{s}'] / name_record.number

        # do final computations
        name_record = name_record.to_dict('records')[0]
        name_record.update({
            'peak_number_year': peak_by_num.year,
            'peak_number': peak_by_num.number,
            'peak_pct_year': peak_by_pct.year,
            'peak_pct': peak_by_pct.pct_year,
            'latest_year': latest.year,
            'latest_number': latest.number,
            'first_appearance': self._first_appearance.loc[self._first_appearance.name.apply(
                lambda x: x.lower()) == name.lower(), 'year'].values[0],
        })
        return name_record

    def add_search_condition(
            self,
            number: tuple = None,
            length: tuple = None,
            start: tuple = None,
            end: tuple = None,
            contains: tuple = None,
            contains_any: tuple = None,
            not_start: tuple = None,
            not_end: tuple = None,
            not_contains: tuple = None,
            order: tuple = None,
            pattern: str = None,
            fem: (bool, tuple) = None,
            masc: (bool, tuple) = None,
            neu: (bool, tuple) = None,
            delta_after: int = None,
            delta_pct: float = None,
            delta_fem_ratio: float = None,
            delta_masc_ratio: float = None,
            after: int = None,
            before: int = None,
    ):
        # set up
        self._after = after
        self._before = before
        df = self.calcd.copy()

        # calculate number/gender delta
        if delta_after:
            if delta_pct is not None:
                df = _calculate_number_delta(df, after=delta_after, pct=delta_pct)
            if delta_fem_ratio is None and delta_masc_ratio is not None:
                delta_fem_ratio = -delta_masc_ratio
            if delta_fem_ratio is not None:
                df = _calculate_gender_delta(df, after=delta_after, fem_ratio=delta_fem_ratio)

        # filter on years
        df = df[df.year.isin(self._years_to_select)].copy()

        # aggregate
        df = df.groupby('name', as_index=False).agg({'number': sum, 'number_f': sum, 'number_m': sum})
        for s in ('f', 'm'):
            df[f'ratio_{s}'] = df[f'number_{s}'] / df.number
        df = df[['name', 'number', 'ratio_f', 'ratio_m']]

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
            return
        df = df.sort_values('number', ascending=False).reset_index(drop=True)
        df = df.iloc[:self._top].copy()
        return df

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


def _calculate_number_delta(df: pd.DataFrame, **delta):
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


def _calculate_gender_delta(df: pd.DataFrame, **delta):
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
