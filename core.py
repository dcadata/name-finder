import os
import re

import numpy as np
import pandas as pd

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
            self._number_per_year, on='year', suffixes=('', '_total'))
        self._name_by_year['pct_year'] = self._name_by_year.number / self._name_by_year.number_total
        self._name_by_year = self._name_by_year.drop(columns='number_total')

        # first appearance
        self._first_appearance = self._raw.groupby('name').year.min()

        # add ratios
        _separate = lambda x: self._raw[self._raw.sex == x].drop(columns='sex')
        self._calcd = _separate('F').merge(_separate('M'), on=['name', 'year'], suffixes=(
            '_f', '_m'), how='outer').merge(self._name_by_year, on=['name', 'year'])
        for s in self._sexes:
            self._calcd[f'number_{s}'] = self._calcd[f'number_{s}'].fillna(0).apply(int)
            # self._calcd[f'ratio_{s}'] = self._calcd[f'number_{s}'] / self._calcd.number

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
        self._delta_cutoff = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000001

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
            historic = df[['year', 'number', 'number_f', 'number_m', 'ratio_f', 'ratio_m']].copy()
            for s in self._sexes:
                historic[f'ratio_{s}'] = historic[f'ratio_{s}'].round(2)

            if n_bars:
                essentially_single_gender = output['ratios']['f'] >= 0.99 or output['ratios']['m'] >= 0.99
                number_bars_mult = 100 / peak.number
                bars_lookback_years = 100
                historic['number_bars'] = (
                        historic.year.apply(str).apply(lambda x: f'{x} ') +
                        historic.number.apply(lambda x: int(round(x * number_bars_mult)) * self._blocks[2] + f' {x:,}')
                )
                historic['ratio_bars'] = (
                        historic.ratio_f.apply(lambda x: 'f ' + int(round(x * 50)) * self._blocks[0]) +
                        historic.ratio_m.apply(lambda x: int(round(x * 50)) * self._blocks[1] + ' m') +
                        historic.year.apply(str).apply(lambda x: f' {x}')
                )
                hist_temp = historic[historic.year.apply(lambda x: (x >= MAX_YEAR - bars_lookback_years) and (x % int(
                    bars_lookback_years / n_bars) == 0))]
                output['display']['number_bars'] = list(hist_temp.number_bars)
                output['display']['ratio_bars'] = [] if essentially_single_gender else list(hist_temp.ratio_bars)

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
            length: tuple = None,
            number_min: int = None,
            number_max: int = None,
            gender: tuple = None,
            after: int = None,
            before: int = None,
            year: int = None,
            delta_after: int = None,
            delta_pct: (float, bool) = None,
            delta_fem: (float, bool) = None,
            top: int = 30,
            as_records: bool = False,
    ) -> dict:
        # set up
        if year:
            after = year
            before = year
        self._after = after
        self._before = before
        df = self._calcd.copy()

        # calculate number/gender delta
        if not delta_after and (delta_pct is not None or delta_fem is not None):  # then use default delta_after
            delta_after = MAX_YEAR - 30

        if delta_after:
            if delta_pct is True:
                delta_pct = self._delta_cutoff
            elif delta_pct is False:
                delta_pct = -self._delta_cutoff

            if delta_fem is True:
                delta_fem = self._delta_cutoff
            elif delta_fem is False:
                delta_fem = -self._delta_cutoff

            if delta_pct is not None:
                df = _calculate_number_delta(df, delta_after, delta_pct)
            if delta_fem is not None:
                df = _calculate_gender_delta(df, delta_after, delta_fem)

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

        df = df.sort_values('number', ascending=False).drop(columns=['name_lower'])
        for s in self._sexes:
            df[f'ratio_{s}'] = df[f'ratio_{s}'].round(2)
        df['display'] = [_create_display_for_search(*i) for i in df[['number', 'ratio_f', 'ratio_m']].to_records(
            index=False)]

        if top:
            df = df.head(top)

        data = df.to_dict('records') if as_records else df
        return data

    def search_by_text(self, query: str, *args, **kwargs) -> dict:
        query = query.lower()
        delta_sections = re.split('trend:', query, 1)
        if len(delta_sections) > 1:
            query, delta_section = delta_sections[0], delta_sections[-1]
        else:
            delta_section = ''

        def _safely_check_regex(pattern: str):
            return _safe_regex_search(pattern, query)

        def _safely_check_regex_and_split_into_tuple(pattern: str):
            result = _safely_check_regex(pattern)
            return tuple(result.split(',')) if result else None

        def _safely_check_regex_delta_section(pattern: str):
            try:
                return re.search(pattern, delta_section).groups()[-1]
            except (AttributeError, IndexError):
                return

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
            delta_pct=dict(down=False, up=True).get(_safely_check_regex_delta_section('(down|up)')),
            delta_fem=dict(fem=True, masc=False).get(_safely_check_regex_delta_section('(fem|masc)')),
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


def _calculate_number_delta(df: pd.DataFrame, after: int, pct: float) -> pd.DataFrame:
    chg = df[df.year == after].merge(df[df.year == MAX_YEAR], on=['name'], suffixes=('_y1', '_y2'))
    if pct > 0:  # trended up
        chg['delta'] = chg.pct_year_y2 >= chg.pct_year_y1 * (1 + pct)
    elif pct < 0:  # trended down
        chg['delta'] = chg.pct_year_y1 >= chg.pct_year_y2 * (1 - pct)
    else:  # no meaningful trend - less than 1% diff
        chg['delta'] = (chg.pct_year_y1 / chg.pct_year_y2).apply(lambda x: 0.99 <= x <= 1.01)
    df = df[df.name.isin(chg[chg.delta].name)].copy()
    return df


def _calculate_gender_delta(df: pd.DataFrame, after: int, fem_ratio: float) -> pd.DataFrame:
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
        short=f'n={number:,}, {display_ratio}',
        info=[
            f'Total Usages: n={number:,} ({numbers_fm})',
            f'Ratio: {display_ratio}',
            f'Peak({peak_year}): n={peak_number:,}',
            f'Latest({latest_year}): n={latest_number:,}',
            f'Earliest({first_appearance})',
        ],
    )


def _create_display_for_search(number: int, ratio_f: float, ratio_m: float) -> str:
    if display_ratio := _create_display_ratio(ratio_f, ratio_m):
        display_ratio = ', ' + display_ratio
    return f'(n={number:,}{display_ratio})'


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
