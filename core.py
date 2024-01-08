import os
import re

import pandas as pd

# years as currently available in dataset
MIN_YEAR = 1880
MAX_YEAR = int(re.search('^yob([0-9]{4}).txt$', os.listdir('data/names/')[-1]).group(1))
PLACEHOLDER_NAMES = ('Unknown', 'Baby', 'Infant')
NEUTRAL_RATIO_RANGE = (.2, .8)


class Filepath:
    NATIONAL_DATA_DIR = 'data/names/'
    TERRITORIES_DATA_DIR = 'data/namesbyterritory/'
    STATES_DATA_DIR = 'data/namesbystate/'
    ACTUARIAL = 'data/actuarial/{sex}.csv'
    AGE_PREDICTION_REFERENCE = 'data/generated/age_prediction_reference.csv'
    GENDER_PREDICTION_REFERENCE = 'data/generated/gender_prediction_reference.csv'
    TOTAL_NUMBER_LIVING_REFERENCE = 'data/generated/raw_with_actuarial.total_number_living.csv'


class Builder:
    def __init__(self, *args, **kwargs) -> None:
        self._national_data_directory = Filepath.NATIONAL_DATA_DIR
        self._territories_data_directory = Filepath.TERRITORIES_DATA_DIR
        self._states_data_directory = Filepath.STATES_DATA_DIR
        self._sexes = ('f', 'm')

    def build_base(self) -> None:
        self._load_data()
        self._transform_data()
        self._load_predict_age_reference()

    def _load_data(self) -> None:
        data = []
        for data_directory, is_territory in [
            (self._national_data_directory, False),
            # (self._territories_data_directory, True),
        ]:
            for filename in os.listdir(data_directory):
                if not filename.lower().endswith('.txt'):
                    continue
                data.append(self._load_one_file(filename, is_territory))
        self._concatenated = pd.concat(data)

    def _load_predict_age_reference(self) -> None:
        self._age_reference = pd.read_csv(
            Filepath.AGE_PREDICTION_REFERENCE, usecols=['name', 'year', 'number_living_pct'],
            dtype=dict(name=str, year=int, number_living_pct=float),
        )

    def _transform_data(self) -> None:
        # combine territories w/ national
        self._raw = self._concatenated.groupby(['name', 'sex', 'year', 'rank_'], as_index=False).number.sum()

        # name by year
        self._name_by_year = self._concatenated.groupby(['name', 'year'], as_index=False).number.sum()

        # add ratios
        _separate_data = lambda x: self._raw[self._raw.sex == x].drop(columns='sex').rename(columns=dict(rank_='rank'))
        self._calcd = _separate_data('F').merge(_separate_data('M'), on=['name', 'year'], suffixes=(
            '_f', '_m'), how='outer').merge(self._name_by_year, on=['name', 'year'])
        for s in self._sexes:
            self._calcd[f'number_{s}'] = self._calcd[f'number_{s}'].fillna(0).map(int)
            self._calcd[f'ratio_{s}'] = self._calcd[f'number_{s}'] / self._calcd.number
            self._calcd[f'rank_{s}'] = self._calcd[f'rank_{s}'].fillna(-1).map(int)

        # add actuarial - loses years before 1900
        self.raw_with_actuarial = self._raw.merge(self._load_actuarial_data(), on=['sex', 'year'])
        self.raw_with_actuarial['number_living'] = (
                self.raw_with_actuarial.number * self.raw_with_actuarial.survival_prob)

    def _load_one_file(self, filename: str, is_territory: bool = None) -> pd.DataFrame:
        df = self._load_one_file_territory(filename) if is_territory else self._load_one_file_national(filename)

        def _add_rank_by_sex(data: pd.DataFrame, sex: str) -> pd.DataFrame:
            data = data[data.sex == sex.upper()].copy()
            data['rank_'] = data.number.rank(method='min', ascending=False)
            return data

        df = pd.concat((_add_rank_by_sex(df, 'f'), _add_rank_by_sex(df, 'm')))
        return df

    def _load_one_file_national(self, filename: str) -> pd.DataFrame:
        dtypes = {'name': str, 'sex': str, 'number': int}
        df = pd.read_csv(self._national_data_directory + filename, names=list(dtypes.keys()), dtype=dtypes).assign(
            year=filename)
        df.year = df.year.apply(lambda x: x.rsplit('.', 1)[0].replace('yob', '')).map(int)
        return df

    def _load_one_file_territory(self, filename: str) -> pd.DataFrame:
        dtypes = {'territory': str, 'sex': str, 'year': int, 'name': str, 'number': int}
        df = pd.read_csv(self._territories_data_directory + filename, names=list(dtypes.keys()), dtype=dtypes).drop(
            columns='territory')
        return df

    def _load_actuarial_data(self) -> pd.DataFrame:
        actuarial = pd.concat(pd.read_csv(Filepath.ACTUARIAL.format(sex=s), usecols=[
            'year', 'age', 'survivors'], dtype=int).assign(sex=s.upper()) for s in self._sexes)
        actuarial = actuarial[actuarial.year == MAX_YEAR].copy()
        actuarial['birth_year'] = actuarial.year - actuarial.age
        actuarial['survival_prob'] = actuarial.survivors / 100_000
        actuarial = actuarial.drop(columns=['year', 'survivors']).rename(columns={'birth_year': 'year'})
        return actuarial

    def _get_state_data(self) -> pd.DataFrame:
        dtypes = {'state': str, 'sex': str, 'year': int, 'name': str, 'number': int}
        df = pd.concat(
            pd.read_csv(self._states_data_directory + filename, names=list(dtypes.keys()), dtype=dtypes)
            for filename in os.listdir(self._states_data_directory) if filename.endswith('.TXT')
        )

        by_name = df[df.year >= 1960].groupby(['name', 'state'], as_index=False).number.sum()
        by_state = df[df.year >= 1960].groupby('state', as_index=False).number.sum()
        data = by_name.merge(by_state, on='state', suffixes=('', '_total'))
        data['pct'] = data.number / data.number_total
        data = data.drop(columns=['number', 'number_total'])
        return data


class Displayer(Builder):
    def __init__(self, *args, **kwargs) -> None:
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
        earliest = df.loc[df.year.idxmin()].copy()

        # filter on years
        df = df[df.year.isin(self.years_to_select)]
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
            'peak': _decompose_peak_or_latest(peak),
            'latest': _decompose_peak_or_latest(latest),
            'earliest': _decompose_peak_or_latest(earliest),
        }
        output['display'] = _create_display_for_name(
            output['numbers']['total'],
            output['numbers']['f'],
            output['numbers']['m'],
            output['ratios']['f'],
            output['ratios']['m'],
            output['peak']['year'],
            output['peak']['numbers']['total'],
            output['latest']['year'],
            output['latest']['numbers']['total'],
        )

        if n_bars:
            historic = df[['year', 'number', 'number_f', 'number_m']].copy()
            for s in self._sexes:
                historic[f'ratio_{s}'] = (historic[f'number_{s}'] / historic.number).round(2)

            essentially_single_gender = output['ratios']['f'] >= 0.99 or output['ratios']['m'] >= 0.99
            number_bars_mult = 100 / peak.number
            historic['number_bars'] = (
                    historic.year.map(str) + ' ' +
                    historic.number.apply(lambda x: int(round(x * number_bars_mult)) * self._blocks[2] + f' {x:,}')
            )
            historic['ratio_bars'] = (
                    'f ' +
                    historic.ratio_f.apply(lambda x: int(round(x * 50)) * self._blocks[0]) +
                    historic.ratio_m.apply(lambda x: int(round(x * 50)) * self._blocks[1]) +
                    ' m ' + historic.year.map(str)
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
            start: tuple = None,
            end: tuple = None,
            contains: tuple = None,
            contains_any: tuple = None,
            not_start: tuple = None,
            not_end: tuple = None,
            not_contains: tuple = None,
            order: tuple = None,
            length: tuple[int, int] = None,
            number_min: int = None,
            number_max: int = None,
            gender: tuple[float, float] | str = None,
            after: int = None,
            before: int = None,
            year: int = None,
            top: int = 20,
            skip: int = None,
            sort_sex: str = None,
            as_records: bool = False,
    ) -> pd.DataFrame | list:
        # set up
        if year:
            after = year
            before = year
        self._after = after
        self._before = before
        df = self._calcd.copy()

        # filter on years
        df = df[df.year.isin(self.years_to_select)].copy()

        # aggregate
        df = df.groupby('name', as_index=False).agg({'number': sum, 'number_f': sum, 'number_m': sum})
        for s in self._sexes:
            df[f'ratio_{s}'] = df[f'number_{s}'] / df.number

        # add lowercase name for filtering
        df['name_lower'] = df.name.str.lower()

        # filter on numbers
        if number_min:
            df = df[df.number >= number_min]
        if number_max:
            df = df[df.number <= number_max]

        # filter on length
        if length:
            df = df[df.name.map(len).apply(lambda x: length[0] <= x <= length[1])]

        # filter on ratio
        if type(gender) == str:
            gender = dict(f=(0, .1), x=NEUTRAL_RATIO_RANGE, m=(.9, 1)).get(gender)
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
            return {}

        sort_field = f'number_{sort_sex}' if sort_sex else 'number'
        df = df.sort_values(sort_field, ascending=False).drop(columns='name_lower')
        for s in self._sexes:
            df[f'ratio_{s}'] = df[f'ratio_{s}'].round(2)

        if as_records:
            df['display'] = [_create_display_for_search(*i) for i in df[[
                'number', 'ratio_f', 'ratio_m']].to_records(index=False)]

        if skip:
            df = df.iloc[skip:]

        if top:
            df = df.head(top)

        data = df.to_dict('records') if as_records else df
        return data

    def search_by_text(self, query: str, *args, **kwargs) -> pd.DataFrame | list:
        query = query.lower()

        def _safely_check_regex(pattern: str) -> str | None:
            return _safe_regex_search(pattern, query)

        def _safely_check_regex_and_split_into_tuple(pattern: str, delimiter: str = ',') -> tuple | None:
            result = _safely_check_regex(pattern)
            if result:
                return tuple(result.split(delimiter))
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

        if gender_ind := _safely_check_regex_and_split_into_tuple('gender:([0-9]+-[0-9]+)', delimiter='-'):
            gender_ind = tuple(map(lambda x: int(x) / 100, gender_ind))

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

    def predict_age(self, name: str, lower_percentile: float) -> dict:
        name = name.title()
        upper_percentile = 1 - lower_percentile
        median_percentile = .5

        data = self._age_reference[self._age_reference.name == name].copy()

        data.number_living_pct = data.number_living_pct.cumsum()
        data['lower'] = (lower_percentile - data.number_living_pct).abs()
        data['upper'] = (upper_percentile - data.number_living_pct).abs()
        data['med'] = (median_percentile - data.number_living_pct).abs()

        data = (
            data.loc[
                (data.lower == data.lower.min()) |
                (data.upper == data.upper.min()) |
                (data.med == data.med.min()),
            ]
            .sort_values('year')
            .assign(bound=['lower', 'median', 'upper'])
            .assign(percentile=[lower_percentile, median_percentile, upper_percentile])
            .set_index('bound')
        )[['percentile', 'year']]

        return dict(
            name=name,
            **data.to_dict('index'),
            most_common=self.get_most_common_year(name),
        )

    def predict_gender(
            self,
            name: str,
            after: int = None,
            before: int = None,
            year: int = None,
            year_band: int = 0,
            living: bool = False,
    ) -> dict:
        # set up
        df = self.raw_with_actuarial.copy()
        output = dict(
            name=name.title(),
            after=after,
            before=before,
            year=year,
            year_band=year_band,
            living=living,
        )

        if living:
            df = df.drop(columns='number').rename(columns={'number_living': 'number'})

        # filter dataframe
        df = df[df['name'].str.lower() == name.lower()].copy()
        if year:
            if year_band:
                years = list(range(year - year_band, year + year_band))
                df = df[df.year.isin(years)]
                output['years'] = years
            else:
                df = df[df.year == year]
        else:
            if after:
                df = df[df.year >= after]
            if before:
                df = df[df.year <= before]

        # add to output
        number = df.number.sum()
        output['number'] = int(number)

        if number:
            numbers = df.groupby('sex').number.sum()
            output.update({
                'prediction': 'f' if numbers.get('F', 0) > numbers.get('M', 0) else 'm',
                'confidence': round(max(numbers.get('F', 0) / number, numbers.get('M', 0) / number), 2),
            })
        return output

    def get_most_common_year(self, name: str) -> dict:
        df = self._calcd.copy()

        # filter on name
        df = df[df['name'].str.lower() == name.lower()]
        if not len(df):
            return {}

        # create metadata df for peak
        peak = df.loc[df.number.idxmax()]
        return dict(year=int(peak.year), number=int(peak.number))

    @property
    def years_to_select(self) -> tuple:
        if self._after and self._before:
            years_range = (self._after, self._before + 1)
        elif self._after:
            years_range = (self._after, MAX_YEAR + 1)
        elif self._before:
            years_range = (MIN_YEAR, self._before + 1)
        else:
            years_range = (MIN_YEAR, MAX_YEAR + 1)
        return tuple(range(*years_range))

    @property
    def calcd(self) -> pd.DataFrame:
        return self._calcd


def _safe_regex_search(pattern: str, text: str) -> str | None:
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
) -> dict:
    numbers_fm = f'f={number_f:,}, m={number_m:,}' if number_f >= number_m else f'm={number_m:,}, f={number_f:,}'
    display_ratio = _create_display_ratio(ratio_f, ratio_m)
    return dict(
        info=[
            f'Total Usages: {number:,} ({numbers_fm})',
            f'Ratio: {display_ratio}',
            f'Peak({peak_year}): {peak_number:,}',
            f'Latest({latest_year}): {latest_number:,}',
        ],
    )


def _create_display_for_search(number: int, ratio_f: float, ratio_m: float) -> str:
    if display_ratio := _create_display_ratio(ratio_f, ratio_m):
        display_ratio = ', ' + display_ratio
    return f'({number:,}{display_ratio})'


def _decompose_peak_or_latest(peak_or_latest: pd.DataFrame) -> dict:
    return dict(
        year=int(peak_or_latest.year),
        numbers=dict(
            total=int(peak_or_latest.number),
            f=int(peak_or_latest.number_f),
            m=int(peak_or_latest.number_m),
        ),
        rank=dict(
            f=int(peak_or_latest.rank_f),
            m=int(peak_or_latest.rank_m)
        ),
    )


def create_predict_gender_reference(
        built_displayer: Displayer = None,
        ages: tuple = None, conf_min: float = .8, n_min: int = 0,
) -> pd.DataFrame:
    if not built_displayer:
        built_displayer = Displayer()
        built_displayer.build_base()

    df = built_displayer.calcd.copy()

    if ages:
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
    df.to_csv(Filepath.GENDER_PREDICTION_REFERENCE, index=False)
    return df


def create_total_number_living_from_actuarial(raw_with_actuarial: pd.DataFrame) -> None:
    total_number_living = raw_with_actuarial.groupby('name', as_index=False).number_living.sum()
    total_number_living.to_csv(Filepath.TOTAL_NUMBER_LIVING_REFERENCE, index=False)


def _read_total_number_living() -> pd.DataFrame:
    total_number_living = pd.read_csv(Filepath.TOTAL_NUMBER_LIVING_REFERENCE, usecols=[
        'name', 'number_living'], dtype=dict(name=str, number_living=float))
    return total_number_living


def create_predict_age_reference(raw_with_actuarial: pd.DataFrame, min_age: int = 0, n_min: int = 0) -> None:
    ref = raw_with_actuarial[['name', 'year', 'age', 'number_living']].copy()
    ref = (
        ref[ref.age >= min_age].drop(columns='age')
        .groupby(['name', 'year'], as_index=False).number_living.sum()  # consolid sex
        .merge(_read_total_number_living(), on='name', suffixes=('', '_name'))
    )
    ref = ref[ref.number_living_name >= n_min].copy()
    ref['number_living_pct'] = ref.number_living / ref.number_living_name
    ref = ref.drop(columns=['number_living', 'number_living_name']).sort_values('year')
    ref.to_csv(Filepath.AGE_PREDICTION_REFERENCE, index=False)


def create_all_generated_data() -> None:
    displayer = Displayer()
    displayer.build_base()

    create_predict_gender_reference(displayer)
    create_total_number_living_from_actuarial(displayer.raw_with_actuarial)
    create_predict_age_reference(displayer.raw_with_actuarial)
