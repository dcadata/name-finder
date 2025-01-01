import os

import pandas as pd

from core import Builder


class StateFilepath:
    NAME_DATA_DIR: str = 'data/namesbystate/'


class StateBuilder(Builder):
    def build_base(self) -> None:
        self._load_name_data()
        return

    def _load_name_data(self) -> None:
        self._raw = pd.concat([
            _load_name_data_for_one_state(filename) for filename in os.listdir(StateFilepath.NAME_DATA_DIR)
            if filename.lower().endswith('.txt')
        ])
        self._raw.sex = self._raw.sex.str.lower()
        self._raw.rank_ = self._raw.rank_.map(int)
        return


def _load_name_data_for_one_state(filename: str) -> pd.DataFrame:
    dtypes = dict(state=str, sex=str, year=int, name=str, number=int)
    df = pd.read_csv(StateFilepath.NAME_DATA_DIR + filename, names=tuple(dtypes.keys()), dtype=dtypes)
    df['rank_'] = df.groupby('sex').number.rank(method='min', ascending=False)
    return df
