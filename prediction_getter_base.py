from time import sleep

import pandas as pd
import requests

_BASE_URL = 'http://127.0.0.1:5000'


class PredictionGetterBase:
    def __init__(self, first_name_field: str, data: pd.DataFrame = None, data_fp: str = None):
        self._first_name_field = first_name_field
        self._data = data
        self._data_fp = data_fp
        self._predictions = []

    def get_predictions(self) -> None:
        self._read_data()
        self._get_predictions()
        self._process_predictions()

    def _read_data(self) -> None:
        if self._data is None and self._data_fp:
            self._data = pd.read_csv(self._data_fp)

    def _get_predictions(self) -> None:
        self._session = requests.Session()
        for first_name in self._data[self._first_name_field].unique():
            self._get_one_prediction(first_name)
        self._session.close()

    def _process_predictions(self) -> None:
        self._predictions = pd.DataFrame(self._predictions).dropna()
        # drop low-confidence predictions
        self._predictions = self._predictions[
            (self._predictions.confidence >= 0.75) &
            (self._predictions.number >= 25)
            ].copy()

    def _get_one_prediction(self, first_name: str) -> None:
        if response := self._session.get(f'{_BASE_URL}/predict/gender/{first_name}', params=dict(living=1)).json():
            temp = dict(
                prediction=response['prediction'],
                confidence=response['confidence'],
                number=response['number'],
            )
            temp.update({self._first_name_field: first_name})
            self._predictions.append(temp)
        sleep(1)
