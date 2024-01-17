import os
import re

import pandas as pd
import seaborn as sns

PLACEHOLDER_NAMES = ('Unknown', 'Baby', 'Infant')
NEUTRAL_RATIO_RANGE = (.2, .8)


class Filepath:
    DATA_DIR = 'data/'
    NATIONAL_DATA_DIR = 'data/names/'
    TERRITORIES_DATA_DIR = 'data/namesbyterritory/'
    ACTUARIAL = 'data/actuarial/{sex}.csv'
    AGE_PREDICTION_REFERENCE = 'data/generated/age_prediction_reference.csv'
    GENDER_PREDICTION_REFERENCE = 'data/generated/gender_prediction_reference.csv'
    TOTAL_NUMBER_LIVING_REFERENCE = 'data/generated/raw_with_actuarial.total_number_living.csv'


class Year:
    MIN_YEAR = 1880
    MAX_YEAR = int(re.search('^yob([0-9]{4}).txt$', os.listdir(Filepath.NATIONAL_DATA_DIR)[-1]).group(1))


class DFAgg:
    NUMBER_SUM = dict(number='sum', number_f='sum', number_m='sum')


class Builder:
    def __init__(self, *args, **kwargs) -> None:
        self._include_territories = kwargs.get('include_territories')
        self._sexes = ('f', 'm')
        self._calcd = None
        self._include_territories = False  # todo: re-rank accounting for territories

    def build_base(self) -> None:
        self._load_data()
        self._build_raw_from_concatenated()
        self._build_peaks()
        self._build_calcd_with_ratios()
        self._build_raw_with_actuarial()
        self._load_predict_age_reference()

    def ingest_alternate_calcd(self, calcd: pd.DataFrame) -> None:
        self._calcd = calcd.copy()

    def _load_data(self) -> None:
        data = []
        for data_directory, is_territory in self._data_directories.items():
            for filename in os.listdir(data_directory):
                if not filename.lower().endswith('.txt'):
                    continue
                data.append(self._load_one_file(filename, is_territory))
        self._concatenated = pd.concat(data)

    def _load_predict_age_reference(self) -> None:
        self._age_reference = pd.read_csv(Filepath.AGE_PREDICTION_REFERENCE, usecols=[
            'name', 'year', 'number_living_pct'], dtype=dict(name=str, year=int, number_living_pct=float))

    def _build_raw_from_concatenated(self) -> None:
        self._concatenated.sex = self._concatenated.sex.str.lower()
        self._concatenated.rank_ = self._concatenated.rank_.map(int)
        if self._include_territories:
            # combine territories w/ national
            self._raw = self._concatenated.groupby(['name', 'sex', 'year', 'rank_'], as_index=False).number.sum()
        else:
            self._raw = self._concatenated.copy()

    def _build_peaks(self) -> None:
        self._peaks = self._raw.groupby(['name', 'sex'], as_index=False).agg(dict(rank_='min')).merge(
            self._raw, on=['name', 'sex', 'rank_'], how='left').sort_values('year')

    def _build_calcd_with_ratios(self) -> None:
        name_by_year = self._raw.groupby(['name', 'year'], as_index=False).number.sum()
        _separate = lambda x: self._raw[self._raw.sex == x].drop(columns='sex').rename(columns=dict(rank_='rank'))
        self._calcd = _separate('f').merge(_separate('m'), on=['name', 'year'], suffixes=(
            '_f', '_m'), how='outer').merge(name_by_year, on=['name', 'year']).sort_values('year')
        for s in self._sexes:
            self._calcd[f'number_{s}'] = self._calcd[f'number_{s}'].fillna(0).map(int)
            self._calcd[f'ratio_{s}'] = self._calcd[f'number_{s}'] / self._calcd.number
            self._calcd[f'rank_{s}'] = self._calcd[f'rank_{s}'].fillna(-1).map(int)

    def _build_raw_with_actuarial(self) -> None:
        # loses years before 1900
        self.raw_with_actuarial = self._raw.merge(self._load_actuarial_data(), on=['sex', 'year'])
        self.raw_with_actuarial['number_living'] = (
                self.raw_with_actuarial.number * self.raw_with_actuarial.survival_prob)

    def _load_one_file(self, filename: str, is_territory: bool = False) -> pd.DataFrame:
        def _add_rank_by_sex(data: pd.DataFrame, sex: str) -> pd.DataFrame:
            data = data[data.sex == sex.upper()].copy()
            data['rank_'] = data.number.rank(method='min', ascending=False)
            return data

        df = self._load_one_file_territory(filename) if is_territory else self._load_one_file_national(filename)
        df = pd.concat((_add_rank_by_sex(df, 'f'), _add_rank_by_sex(df, 'm')))
        return df

    @staticmethod
    def _load_one_file_national(filename: str) -> pd.DataFrame:
        year = re.search('yob([0-9]+)\.txt', filename).group(1)
        dtypes = {'name': str, 'sex': str, 'number': int}
        df = pd.read_csv(Filepath.NATIONAL_DATA_DIR + filename, names=list(dtypes.keys()), dtype=dtypes).assign(
            year=year)
        df.year = df.year.map(int)
        return df

    @staticmethod
    def _load_one_file_territory(filename: str) -> pd.DataFrame:
        dtypes = {'territory': str, 'sex': str, 'year': int, 'name': str, 'number': int}
        df = pd.read_csv(Filepath.TERRITORIES_DATA_DIR + filename, names=list(dtypes.keys()), dtype=dtypes).drop(
            columns='territory')
        return df

    def _load_actuarial_data(self) -> pd.DataFrame:
        actuarial = pd.concat(pd.read_csv(Filepath.ACTUARIAL.format(sex=s), usecols=[
            'year', 'age', 'survivors'], dtype=int).assign(sex=s) for s in self._sexes)
        actuarial = actuarial[actuarial.year == Year.MAX_YEAR].copy()
        actuarial['birth_year'] = actuarial.year - actuarial.age
        actuarial['survival_prob'] = actuarial.survivors / 100_000
        actuarial = actuarial.drop(columns=['year', 'survivors']).rename(columns={'birth_year': 'year'})
        return actuarial

    @property
    def _data_directories(self) -> dict:
        data_directories = {Filepath.NATIONAL_DATA_DIR: False}
        if self._include_territories:
            data_directories[Filepath.TERRITORIES_DATA_DIR] = True
        return data_directories

    @property
    def calculated(self) -> pd.DataFrame:
        return self._calcd


class Displayer(Builder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._after = None
        self._before = None

    def name(
            self,
            name: str,
            after: int = None,
            before: int = None,
            year: int = None,
            display: bool = None,
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

        # build metadata
        earliest, latest = df.iloc[[0, -1]].to_dict('records')

        # filter on years
        df = df[df.year.isin(self.years_to_select)]
        if not len(df):
            return {}

        # aggregate
        grouped = df.groupby('name', as_index=False).agg(DFAgg.NUMBER_SUM)
        for s in self._sexes:
            grouped[f'ratio_{s}'] = (grouped[f'number_{s}'] / grouped.number).round(2)

        # build output
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
            'peak': self._get_peak(name),
            'latest': _restructure_earliest_or_latest(latest),
            'earliest': _restructure_earliest_or_latest(earliest),
        }

        if display:
            _make_plot_for_name(df, name.title())

        return output

    def search(
            self,
            pattern: str = None,
            start: tuple[str, ...] = None,
            end: tuple[str, ...] = None,
            contains: tuple[str, ...] = None,
            contains_any: tuple[str, ...] = None,
            not_start: tuple[str, ...] = None,
            not_end: tuple[str, ...] = None,
            not_contains: tuple[str, ...] = None,
            order: tuple[str, ...] = None,
            length_min: int = None,
            length_max: int = None,
            number_min: int = None,
            number_max: int = None,
            gender: tuple[float, float] | str = None,
            after: int = None,
            before: int = None,
            year: int = None,
            peaked: bool = False,
            rank: int = None,
            top: int = 20,
            skip: int = None,
            sort_sex: str = None,
            display: bool = False,
    ) -> pd.DataFrame | list[str, ...]:
        # set up
        if year:
            after = year
            before = year
        self._after = after
        self._before = before
        if rank:
            skip = rank - 1
            top = 1
        df = self._calcd.copy()

        # filter on years
        df = df[df.year.isin(self.years_to_select)].copy()

        # aggregate
        agg_fields = DFAgg.NUMBER_SUM.copy()
        if len(self.years_to_select) == 1:
            agg_fields.update(dict(rank_f='min', rank_m='min'))
        df = df.groupby('name', as_index=False).agg(agg_fields)
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
        if length_min or length_max:
            if length_min:
                df = df[df.name.map(len) >= length_min]
            if length_max:
                df = df[df.name.map(len) <= length_max]

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
            return df

        sort_field = f'number_{sort_sex}' if sort_sex else 'number'
        df = df.sort_values(sort_field, ascending=False).drop(columns='name_lower')
        for s in self._sexes:
            df[f'ratio_{s}'] = df[f'ratio_{s}'].round(2)

        if peaked:
            peaked_within = self._peaks[(self._peaks.year >= self.after) & (self._peaks.year <= self.before)]
            df = df[df.name.isin(peaked_within.name)].copy()

        if skip:
            df = df.iloc[skip:].copy()

        if top:
            df = df.head(top).copy()

        if display:
            return [_make_search_display_string(*i) for i in df[['name', 'number', 'ratio_f', 'ratio_m']].to_records(
                index=False)]
        return df

    def predict_age(self, name: str, mid_percentile: float = .68) -> pd.DataFrame:
        name = name.title()
        lower_percentile = .5 - mid_percentile / 2
        upper_percentile = 1 - lower_percentile

        df = self._age_reference[self._age_reference.name == name].copy()

        df.number_living_pct = df.number_living_pct.cumsum()
        df['lower'] = (lower_percentile - df.number_living_pct).abs()
        df['upper'] = (upper_percentile - df.number_living_pct).abs()

        df = (
            df.loc[(df.lower == df.lower.min()) | (df.upper == df.upper.min())]
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
            living: bool = False,
    ) -> dict:
        # set up
        df = self.raw_with_actuarial.copy()
        output = dict(name=name.title())

        if living:
            df = df.drop(columns='number').rename(columns={'number_living': 'number'})
            output['living'] = True

        # filter dataframe
        df = df[df['name'].str.lower() == name.lower()].copy()
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
            prediction = 'f' if numbers.f > numbers.m else 'm'
            output.update(dict(
                prediction=prediction,
                confidence=round(numbers[prediction] / number, 2),
            ))

        return output

    def _get_peak(self, name: str) -> dict:
        return self._peaks[self._peaks.name == name.title()].drop_duplicates(subset=['sex'], keep='last').rename(
            columns=dict(rank_='rank')).set_index('sex')[['year', 'rank', 'number']].to_dict('index')

    @property
    def after(self) -> int:
        return self.years_to_select[0]

    @property
    def before(self) -> int:
        return self.years_to_select[-1]

    @property
    def years_to_select(self) -> tuple[int, ...]:
        if self._after and self._before:
            years_range = (self._after, self._before + 1)
        elif self._after:
            years_range = (self._after, Year.MAX_YEAR + 1)
        elif self._before:
            years_range = (Year.MIN_YEAR, self._before + 1)
        else:
            years_range = (Year.MIN_YEAR, Year.MAX_YEAR + 1)
        return tuple(range(*years_range))


def _make_plot_for_name(filt_df: pd.DataFrame, name: str) -> None:
    historic = filt_df[['year', 'number_f', 'number_m']].melt(['year'], ['number_f', 'number_m'], 'gender', 'number')
    historic.gender = historic.gender.str.slice(-1)
    ax = sns.lineplot(historic, x='year', y='number', hue='gender', palette=('red', 'blue'), hue_order=('f', 'm'))
    ax.set_title(name)
    ax.figure.tight_layout()


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


def build_predict_gender_reference(
        displayer: Displayer = None,
        after: int = None,
        before: int = None,
        conf_min: float = .8,
        n_min: int = 0,
) -> None:
    df = displayer.calculated.copy()

    if after:
        df = df[df.year >= after]
    if before:
        df = df[df.year <= before]

    df = df.groupby('name', as_index=False).agg(DFAgg.NUMBER_SUM)

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

    df[['name', 'gender_prediction']].to_csv(Filepath.GENDER_PREDICTION_REFERENCE, index=False)


def build_total_number_living_from_actuarial(raw_with_actuarial: pd.DataFrame) -> None:
    total_number_living = raw_with_actuarial.groupby('name', as_index=False).number_living.sum()
    total_number_living.to_csv(Filepath.TOTAL_NUMBER_LIVING_REFERENCE, index=False)


def _read_total_number_living() -> pd.DataFrame:
    total_number_living = pd.read_csv(Filepath.TOTAL_NUMBER_LIVING_REFERENCE, usecols=[
        'name', 'number_living'], dtype=dict(name=str, number_living=float))
    return total_number_living


def build_predict_age_reference(raw_with_actuarial: pd.DataFrame, min_age: int = 0, n_min: int = 0) -> None:
    ref = raw_with_actuarial[['name', 'year', 'age', 'number_living']].copy()
    ref = (
        ref[ref.age >= min_age].drop(columns='age')
        .groupby(['name', 'year'], as_index=False).number_living.sum()
        .merge(_read_total_number_living(), on='name', suffixes=('', '_name'))
    )
    ref = ref[ref.number_living_name >= n_min].copy()
    ref['number_living_pct'] = ref.number_living / ref.number_living_name
    ref = ref.drop(columns=['number_living', 'number_living_name']).sort_values('year')
    ref.to_csv(Filepath.AGE_PREDICTION_REFERENCE, index=False)


def build_all_generated_data() -> None:
    displayer = Displayer()
    displayer.build_base()

    build_predict_gender_reference(displayer)
    build_total_number_living_from_actuarial(displayer.raw_with_actuarial)
    build_predict_age_reference(displayer.raw_with_actuarial)
