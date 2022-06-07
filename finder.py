import os
import re

import numpy as np
import pandas as pd

# years as currently available in dataset
_MIN_YEAR = 1880
MAX_YEAR = int(re.search('^yob([0-9]{4}).txt$', os.listdir('data/names/')[-1]).group(1))
OUTPUT_RECORDS = True


class Loader:
    def __init__(self, *args, **kwargs):
        self._national_data_directory = 'data/names/'
        self._territories_data_directory = 'data/namesbyterritory/'

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
        actuarial = pd.read_csv(f'data/actuarial/{sex}.csv', usecols=['year', 'age', 'survivors'], dtype=int)
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
        self._delta_cutoff = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000001

    def name(
            self,
            name: str,
            after: int = None,
            before: int = None,
            year: int = None,
            show_historic: bool = None,
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
        for s in ('f', 'm'):
            grouped[f'ratio_{s}'] = (grouped[f'number_{s}'] / grouped.number).apply(lambda x: round(x, 2))

        # create output
        grouped = grouped.to_dict('records')[0]
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
        # noinspection PyTypeChecker
        output['display'] = _create_display_for_name(
            output['name'],
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

        if show_historic:
            historic = df[['year', 'number', 'number_f', 'number_m', 'ratio_f', 'ratio_m']].copy()
            for s in ('f', 'm'):
                historic[f'ratio_{s}'] = historic[f'ratio_{s}'].apply(lambda x: round(x, 2))
            output['historic'] = list(historic.to_dict('records')) if OUTPUT_RECORDS else historic
        return output

    def compare(self, names: tuple, *args, **kwargs) -> dict:
        data = [self.name(name, *args, **kwargs) for name in names]
        data = dict(data=data, display='\n\n'.join(i['display'] for i in data))
        return data

    def name_or_compare_by_text(self, text: str) -> dict:
        text = text.lower()

        def _safely_check_regex(pattern: str):
            return _safe_regex_search(pattern, text)

        names_ind = text.split(None, 1)[0].split('/')
        after_ind = _safely_check_regex('(after|since)\s([0-9]{4})')
        before_ind = _safely_check_regex('before\s([0-9]{4})')
        year_ind = _safely_check_regex('in\s([0-9]{4})')

        conditions = dict(
            after=int(after_ind) if after_ind else None,
            before=int(before_ind) if before_ind else None,
            year=int(year_ind) if year_ind else None,
        )
        if len(names_ind) == 1:
            conditions['name'] = names_ind[0]
            data = self.name(**conditions)
        else:
            conditions['names'] = tuple(names_ind)
            data = self.compare(**conditions)
        data['conditions'] = conditions
        return data

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
            delta_after = MAX_YEAR - 20

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
            return {}

        df = df.sort_values('number', ascending=False).drop(columns=['name_lower'])
        for s in ('f', 'm'):
            df[f'ratio_{s}'] = df[f'ratio_{s}'].apply(lambda x: round(x, 2))
        df['display'] = [_create_display_for_search(*i) for i in df[[
            'name', 'number', 'ratio_f', 'ratio_m']].to_records(index=False)]

        if top:
            df = df.head(top)

        if OUTPUT_RECORDS:
            records = df.to_dict('records')
            data = dict(data=records, display=', '.join(i['display'] for i in records))
        else:
            data = dict(data=df, display=', '.join(df.display))
        return data

    def search_by_text(self, text: str) -> dict:
        text = text.lower()
        delta_sections = re.split('trend(ing|ed)?\s', text, 1)
        if len(delta_sections) > 1:
            text, delta_section = delta_sections[0], delta_sections[-1]
        else:
            delta_section = ''

        def _safely_check_regex(pattern: str):
            return _safe_regex_search(pattern, text)

        def _safely_check_regex_and_split_into_tuple(pattern: str):
            result = _safely_check_regex(pattern)
            return tuple(result.split(',')) if result else None

        def _safely_check_regex_delta_section(pattern: str):
            try:
                return re.search(pattern, delta_section).groups()[-1]
            except (AttributeError, IndexError):
                return

        length_ind = _safely_check_regex('(short|med|long)')
        gender_ind = _safely_check_regex('(fem|neu|unisex|masc)')
        after_ind = _safely_check_regex('(after|since)\s([0-9]{4})')
        before_ind = _safely_check_regex('before\s([0-9]{4})')
        year_ind = _safely_check_regex('in\s([0-9]{4})')
        delta_after_ind = _safely_check_regex_delta_section('(after|since)\s([0-9]{4})')
        delta_pct_ind = _safely_check_regex_delta_section('(down|up)')
        delta_gender_ind = _safely_check_regex_delta_section('(fem|masc)')

        conditions = dict(
            pattern=_safely_check_regex('(pattern|regex)\s(.*)'),
            start=_safely_check_regex_and_split_into_tuple('(start|beginn?)(ing|s)?(\swith)?\s([a-z,]+)'),
            end=_safely_check_regex_and_split_into_tuple('end(ing|s)?(\swith)?\s([a-z,]+)'),
            contains=_safely_check_regex_and_split_into_tuple('contain(ing|s)?\s([a-z,]+)'),
            order=_safely_check_regex_and_split_into_tuple('order\s([a-z,]+)'),
            length=dict(short=(3, 5), med=(6, 8), long=(9, 30)).get(length_ind),
            gender=dict(fem=(0, 0.2), neu=(0.2, 0.8), unisex=(0.2, 0.8), masc=(0.8, 1)).get(gender_ind),
            after=int(after_ind) if after_ind else None,
            before=int(before_ind) if before_ind else None,
            year=int(year_ind) if year_ind else None,
            delta_after=int(delta_after_ind) if delta_after_ind else None,
            delta_pct=dict(down=-self._delta_cutoff, up=self._delta_cutoff).get(delta_pct_ind),
            delta_fem=dict(fem=self._delta_cutoff, masc=-self._delta_cutoff).get(delta_gender_ind),
        )
        data = self.search(**conditions)
        data['conditions'] = conditions
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


def _create_display_ratio(ratio_f: float, ratio_m: float, ignore_ones: bool = True) -> str:
    if ignore_ones and (ratio_f == 1 or ratio_m == 1):
        return ''
    elif ratio_f > ratio_m:
        return f'f={ratio_f}'
    elif ratio_m > ratio_f:
        return f'm={ratio_m}'
    else:  # they're equal
        return 'no lean'


def _create_display_for_name(
        name: str,
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
) -> str:
    numbers_fm = f'f={number_f:,}, m={number_m:,}' if number_f >= number_m else f'm={number_m:,}, f={number_f:,}'
    sections = (
        name,
        f'. Total Usages: n={number:,} ({numbers_fm})',
        f'. Ratio: {_create_display_ratio(ratio_f, ratio_m, ignore_ones=False)}',
        f'. Peak({peak_year}): n={peak_number:,}',
        f'. Latest({latest_year}): n={latest_number:,}',
        f'. Earliest({first_appearance})',
    )
    result = '  \n'.join(sections)
    return result


def _create_display_for_search(name: str, number: int, ratio_f: float, ratio_m: float) -> str:
    if display_ratio := _create_display_ratio(ratio_f, ratio_m):
        display_ratio = ', ' + display_ratio
    return f'{name}(n={number:,}{display_ratio})'


def _create_predict_gender_reference(calcd: pd.DataFrame) -> pd.DataFrame:
    calcd = calcd[calcd.year.apply(lambda x: MAX_YEAR - 80 <= x <= MAX_YEAR - 25)].drop(columns=[
        'ratio_f', 'ratio_m', 'pct_year'])
    calcd = calcd.groupby('name', as_index=False).agg(dict(number=sum, number_f=sum, number_m=sum))
    for s in ('f', 'm'):
        calcd[f'ratio_{s}'] = calcd[f'number_{s}'] / calcd.number
    calcd.loc[calcd.ratio_f > calcd.ratio_m, 'gender_prediction'] = 'f'
    calcd.loc[calcd.ratio_f < calcd.ratio_m, 'gender_prediction'] = 'm'
    calcd.loc[calcd.ratio_f == calcd.ratio_m, 'gender_prediction'] = 'x'
    return calcd[['name', 'gender_prediction']]
