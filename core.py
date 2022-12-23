import os
import re

import numpy as np
import pandas as pd
from scipy import stats

# years as currently available in dataset
_MIN_YEAR = 1880
MAX_YEAR = int(re.search('^yob([0-9]{4}).txt$', os.listdir('data/names/')[-1]).group(1))


class Loader:
    def __init__(self, *args, **kwargs):
        self._national_data_directory = 'data/names/'
        self._territories_data_directory = 'data/namesbyterritory/'
        self._sexes = ('f', 'm')

    def load(self) -> None:
        self._read_data()
        self._transform()

    def _read_data(self) -> None:
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

    def _transform(self) -> None:
        # combine territories w/ national
        self._raw = self._concatenated.groupby(['name', 'sex', 'year'], as_index=False).number.sum()
        self._number_per_year = self._concatenated.groupby('year', as_index=False).number.sum()

        # name by year
        self._name_by_year = self._concatenated.groupby(['name', 'year'], as_index=False).number.sum().merge(
            self._number_per_year, on='year', suffixes=('', '_total')).drop(columns='number_total')

        # first appearance
        self._first_appearance = self._raw.groupby('name').year.min()

        # add ratios
        _separate = lambda x: self._raw[self._raw.sex == x].drop(columns='sex')
        self._calcd = _separate('F').merge(_separate('M'), on=['name', 'year'], suffixes=(
            '_f', '_m'), how='outer').merge(self._name_by_year, on=['name', 'year'])
        for s in self._sexes:
            self._calcd[f'number_{s}'] = self._calcd[f'number_{s}'].fillna(0).apply(int)
            self._calcd[f'ratio_{s}'] = self._calcd[f'number_{s}'] / self._calcd.number

        # add actuarial - loses years before 1900
        self._raw_with_actuarial = self._raw.merge(self._read_actuarial_data(), on=['sex', 'year'])
        self._raw_with_actuarial['number_living'] = (
                self._raw_with_actuarial.number * self._raw_with_actuarial.survival_prob)

    def _read_one_file(self, filename: str, is_territory: bool = None) -> pd.DataFrame:
        df = self._read_one_file_territory(filename) if is_territory else self._read_one_file_national(filename)
        return df

    def _read_one_file_national(self, filename: str) -> pd.DataFrame:
        dtypes = {'name': str, 'sex': str, 'number': int}
        df = pd.read_csv(self._national_data_directory + filename, names=list(dtypes.keys()), dtype=dtypes).assign(
            year=filename)
        df.year = df.year.apply(lambda x: x.rsplit('.', 1)[0].replace('yob', '')).apply(int)
        return df

    def _read_one_file_territory(self, filename: str) -> pd.DataFrame:
        dtypes = {'territory': str, 'sex': str, 'year': int, 'name': str, 'number': int}
        df = pd.read_csv(self._territories_data_directory + filename, names=list(dtypes.keys()), dtype=dtypes).drop(
            columns='territory')
        return df

    def _read_actuarial_data(self) -> pd.DataFrame:
        actuarial = pd.concat(pd.read_csv(f'data/actuarial/{s}.csv', usecols=[
            'year', 'age', 'survivors'], dtype=int).assign(sex=s.upper()) for s in self._sexes)
        actuarial = actuarial[actuarial.year == MAX_YEAR].copy()
        actuarial['birth_year'] = actuarial.year - actuarial.age
        actuarial['survival_prob'] = actuarial.survivors.apply(lambda x: x / 100_000)
        actuarial = actuarial.drop(columns=['year', 'survivors']).rename(columns={'birth_year': 'year'})
        return actuarial


class Displayer(Loader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._after = None
        self._before = None
        self.number_bars_header_text = 'Number of Usages (scaled)'
        self.ratio_bars_header_text = 'Gender Ratio (f <-> m)'
        self._blocks = '▓', '▒', '░'

    def name(
            self,
            name: str,
            after: int = None,
            before: int = None,
            year: int = None,
            n_bars: int = None,
    ) -> dict:
        # set up
        if year:
            after = year
            before = year
        self._after = after
        self._before = before
        df = self._calcd.copy()

        # filter on name
        df = df[df['name'].str.lower() == name.lower()]
        if not len(df):
            return {}

        # create metadata dfs
        peak = df.loc[df.number.idxmax()].copy()
        latest = df.loc[df.year.idxmax()].copy()

        # filter on years
        df = df[df.year.isin(self._years_to_select)]
        if not len(df):
            return {}

        df = df.sort_values('year')

        # aggregate
        grouped = df.groupby('name', as_index=False).agg({'number': sum, 'number_f': sum, 'number_m': sum})
        for s in self._sexes:
            grouped[f'ratio_{s}'] = (grouped[f'number_{s}'] / grouped.number).round(2)

        # create output
        grouped = grouped.iloc[0].to_dict()
        output = {
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
                'year': int(peak.year),
                'number': int(peak.number),
            },
            'latest': {
                'year': int(latest.year),
                'number': int(latest.number),
            },
            'first_appearance': int(self._first_appearance[grouped['name']]),
        }
        output['display'] = _create_display_for_name(
            output['numbers']['total'],
            output['numbers']['f'],
            output['numbers']['m'],
            output['ratios']['f'],
            output['ratios']['m'],
            output['peak']['year'],
            output['peak']['number'],
            output['latest']['year'],
            output['latest']['number'],
            output['first_appearance'],
        )

        if n_bars:
            historic = df[['year', 'number', 'number_f', 'number_m']].copy()
            for s in self._sexes:
                historic[f'ratio_{s}'] = (historic[f'number_{s}'] / historic.number).round(2)

            if len(historic) > 1:
                number_linreg = stats.linregress(historic.year, historic.number.apply(lambda x: x * 100))
                if number_linreg.pvalue < 0.05:
                    output['number_trend'] = round(number_linreg.slope, 2)
                ratio_f_linreg = stats.linregress(historic.year, historic.ratio_f.apply(lambda x: x * 100))
                if ratio_f_linreg.pvalue < 0.05:
                    output['ratio_f_trend'] = round(ratio_f_linreg.slope, 2)

            essentially_single_gender = output['ratios']['f'] >= 0.99 or output['ratios']['m'] >= 0.99
            number_bars_mult = 100 / peak.number
            historic['number_bars'] = (
                    historic.year.apply(str).apply(lambda x: f'{x} ') +
                    historic.number.apply(lambda x: int(round(x * number_bars_mult)) * self._blocks[2] + f' {x:,}')
            )
            historic['ratio_bars'] = (
                    historic.ratio_f.apply(lambda x: 'f ' + int(round(x * 50)) * self._blocks[0]) +
                    historic.ratio_m.apply(lambda x: int(round(x * 50)) * self._blocks[1] + ' m') +
                    historic.year.apply(str).apply(lambda x: f' {x}')
            )
            if n_bars == -1:
                hist_temp = historic  # show one bar per year
            else:
                hist_temp = historic[historic.year.apply(lambda x: x % int(100 / n_bars) == 0)]
            output['display']['number_bars'] = hist_temp.number_bars.to_list()
            output['display']['ratio_bars'] = [] if essentially_single_gender else hist_temp.ratio_bars.to_list()

        return output

    def search(
            self,
            pattern: str = None,
            start: tuple[str] = None,
            end: tuple[str] = None,
            contains: tuple[str] = None,
            contains_any: tuple[str] = None,
            not_start: tuple[str] = None,
            not_end: tuple[str] = None,
            not_contains: tuple[str] = None,
            order: tuple[str] = None,
            length: tuple[int] = None,
            number_min: int = None,
            number_max: int = None,
            gender: tuple[float] = None,
            after: int = None,
            before: int = None,
            year: int = None,
            top: int = 30,
            as_records: bool = False,
    ) -> (list, pd.DataFrame):
        # set up
        if year:
            after = year
            before = year
        self._after = after
        self._before = before
        df = self._calcd.copy()

        # filter on years
        df = df[df.year.isin(self._years_to_select)].copy()

        # aggregate
        df = df.groupby('name', as_index=False).agg({'number': sum, 'number_f': sum, 'number_m': sum})
        for s in self._sexes:
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
            return {}

        df = df.sort_values('number', ascending=False).drop(columns='name_lower')
        for s in self._sexes:
            df[f'ratio_{s}'] = df[f'ratio_{s}'].round(2)
        df['display'] = [_create_display_for_search(*i) for i in df[['number', 'ratio_f', 'ratio_m']].to_records(
            index=False)]

        if top:
            df = df.head(top)

        data = df.to_dict('records') if as_records else df
        return data

    def search_by_text(self, query: str, *args, **kwargs) -> (list, pd.DataFrame):
        query = query.lower()

        def _safely_check_regex(pattern: str):
            return _safe_regex_search(pattern, query)

        def _safely_check_regex_and_split_into_tuple(pattern: str):
            result = _safely_check_regex(pattern)
            return tuple(result.split(',')) if result else None

        length_ind = _safely_check_regex('length:([0-9]+-[0-9]+)')
        year_ind = _safely_check_regex('year:([0-9]{4})')
        if years_ind := re.search('years:([0-9]{4})-([0-9]{4})', query, re.I):
            after_ind, before_ind = map(int, years_ind.groups())
        else:
            if after_ind := _safely_check_regex('after:([0-9]{4})'):
                after_ind = int(after_ind)
            if before_ind := _safely_check_regex('before:([0-9]{4})'):
                before_ind = int(before_ind)

        if gender_ind := _safely_check_regex_and_split_into_tuple('gender:([fxm,]{1,3})'):
            gender_ind = set(gender_ind)
            if gender_ind == {'f', 'x'}:
                gender_ind = (0, 0.8)
            elif gender_ind == {'m', 'x'}:
                gender_ind = (0.2, 1)
            elif len(gender_ind) == 1:
                gender_ind = dict(f=(0, 0.2), x=(0.2, 0.8), m=(0.8, 1)).get(gender_ind.pop())

        data = self.search(
            pattern=_safely_check_regex('pattern:(.*)'),
            start=_safely_check_regex_and_split_into_tuple('(\s|^)start:([a-z,]+)'),
            end=_safely_check_regex_and_split_into_tuple('(\s|^)end:([a-z,]+)'),
            contains=_safely_check_regex_and_split_into_tuple('(\s|^)contains:([a-z,]+)'),
            contains_any=_safely_check_regex_and_split_into_tuple('(\s|^)contains-any:([a-z,]+)'),
            not_start=_safely_check_regex_and_split_into_tuple('~start:([a-z,]+)'),
            not_end=_safely_check_regex_and_split_into_tuple('~end:([a-z,]+)'),
            not_contains=_safely_check_regex_and_split_into_tuple('~contains:([a-z,]+)'),
            order=_safely_check_regex_and_split_into_tuple('order:([a-z,]+)'),
            length=tuple(map(int, length_ind.split('-'))) if length_ind else None,
            gender=gender_ind,
            after=after_ind,
            before=before_ind,
            year=int(year_ind) if year_ind else None,
            *args,
            **kwargs,
        )
        return data

    def predict_age(
            self,
            name: str,
            gender: str = None,
            living: bool = False,
            buckets: int = None,
    ) -> dict:
        # set up
        output = {}
        df = self._raw_with_actuarial.copy()
        if living:
            # noinspection PyArgumentList
            df = df.drop(columns=['number']).rename(columns={'number_living': 'number'})
            output['living'] = True

        # filter dataframe
        df = df[df['name'].str.lower() == name.lower()].copy()
        if gender:
            df = df[df.sex == gender.upper()].copy()
            output['gender'] = gender.upper()

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

        # add to output
        output.update({'name': name.title(), 'number': int(number)})
        if buckets:
            output['buckets'] = buckets
        output['prediction'] = prediction
        return output

    def predict_gender(
            self,
            name: str,
            after: int = None,
            before: int = None,
            year: int = None,
            living: bool = False,
    ) -> dict:
        # set up
        output = {}
        df = self._raw_with_actuarial.copy()
        if living:
            # noinspection PyArgumentList
            df = df.drop(columns=['number']).rename(columns={'number_living': 'number'})
            output['living'] = True

        # filter dataframe
        df = df[df['name'].str.lower() == name.lower()].copy()
        if year:
            birth_years = list(range(year - 2, year + 3))
            df = df[df.year.isin(birth_years)]
            output['birth_year_range'] = birth_years
        else:
            if after:
                df = df[df.year >= after]
                output['after'] = after
            if before:
                df = df[df.year <= before]
                output['before'] = before

        if not len(df):
            return {}

        # add to output
        number = df.number.sum()
        if number < 25:
            return {}

        numbers = df.groupby('sex').number.sum()
        output.update({
            'name': name.title(),
            'number': int(number),
            'prediction': 'F' if numbers.get('F', 0) > numbers.get('M', 0) else 'M',
            'confidence': round(max(numbers.get('F', 0) / number, numbers.get('M', 0) / number), 2),
        })
        return output

    @property
    def _years_to_select(self) -> tuple:
        if self._after and self._before:
            years_range = (self._after, self._before + 1)
        elif self._after:
            years_range = (self._after, MAX_YEAR + 1)
        elif self._before:
            years_range = (_MIN_YEAR, self._before + 1)
        else:
            years_range = (_MIN_YEAR, MAX_YEAR + 1)
        return tuple(range(*years_range))


def _safe_regex_search(pattern: str, text: str):
    try:
        return re.search(pattern, text).groups()[-1]
    except (AttributeError, IndexError):
        return


def _create_display_ratio(ratio_f: float, ratio_m: float, ignore_ones: bool = False) -> str:
    if ignore_ones and (ratio_f == 1 or ratio_m == 1):
        return ''
    elif ratio_f > ratio_m:
        return f'f={ratio_f}'
    elif ratio_m > ratio_f:
        return f'm={ratio_m}'
    else:  # they're equal
        return 'no lean'


def _create_display_for_name(
        number: int,
        number_f: int,
        number_m: int,
        ratio_f: float,
        ratio_m: float,
        peak_year: int,
        peak_number: int,
        latest_year: int,
        latest_number: int,
        first_appearance: int,
) -> dict:
    numbers_fm = f'f={number_f:,}, m={number_m:,}' if number_f >= number_m else f'm={number_m:,}, f={number_f:,}'
    display_ratio = _create_display_ratio(ratio_f, ratio_m)
    return dict(
        info=[
            f'Total Usages: {number:,} ({numbers_fm})',
            f'Ratio: {display_ratio}',
            f'Peak({peak_year}): {peak_number:,}',
            f'Latest({latest_year}): {latest_number:,}',
            f'Earliest({first_appearance})',
        ],
    )


def _create_display_for_search(number: int, ratio_f: float, ratio_m: float) -> str:
    if display_ratio := _create_display_ratio(ratio_f, ratio_m):
        display_ratio = ', ' + display_ratio
    return f'({number:,}{display_ratio})'


def create_predict_gender_reference(ages: tuple = (18, 80), conf_min: float = 0.7, n_min: int = 10) -> pd.DataFrame:
    displayer = Displayer()
    displayer.load()
    df = displayer._calcd.copy()

    df = df[df.year.apply(lambda x: MAX_YEAR - ages[1] <= x <= MAX_YEAR - ages[0])].copy()

    df = df.groupby('name', as_index=False).agg(dict(number=sum, number_f=sum, number_m=sum))

    df.loc[df.number_f > df.number_m, 'gender_prediction'] = 'f'
    df.loc[df.number_f < df.number_m, 'gender_prediction'] = 'm'
    df.loc[df.number_f == df.number_m, 'gender_prediction'] = 'x'

    if conf_min:
        ratio_f = df.number_f / df.number
        ratio_m = df.number_m / df.number
        df.loc[(ratio_f < conf_min) & (ratio_m < conf_min), 'gender_prediction'] = 'x'

    if n_min:
        df.loc[df.number < n_min, 'gender_prediction'] = 'rare'

    df.gender_prediction = df.gender_prediction.fillna('unk')

    df = df[['name', 'gender_prediction']].copy()
    df.to_csv('gender_prediction_reference.csv', index=False)
    return df
