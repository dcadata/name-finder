import zipfile
from time import sleep

import pandas as pd
import requests

from core import Year, Filepath, create_all_generated_data


class SsaDataRefresher:
    def __init__(self) -> None:
        self.refreshed = False

    def refresh(self) -> None:
        self._open_session()
        self.refreshed = self._refresh_name_data()
        if self.refreshed:
            self._refresh_actuarial_data()
        self._close_session()

    def _open_session(self) -> None:
        self._session = requests.Session()

    def _close_session(self) -> None:
        self._session.close()

    def _refresh_name_data(self) -> bool:
        # compare to website
        table = pd.read_html(self._session.get('https://www.ssa.gov/oact/babynames/limits.html').text)[0]
        if Year.MAX_YEAR >= int(table.iloc[0, 0]):
            return False

        sleep(3)
        # if year has been added, then download files
        for url in (
                'https://www.ssa.gov/oact/babynames/names.zip',
                'https://www.ssa.gov/oact/babynames/territory/namesbyterritory.zip',
        ):
            filepath = 'data/' + url.rsplit('/', 1)[1]
            with open(filepath, 'wb') as f:
                f.write(self._session.get(url).content)
            sleep(3)
            with zipfile.ZipFile(filepath) as z:
                z.extractall(filepath[:-4])
        return True

    def _refresh_actuarial_data(self) -> None:
        url = 'https://www.ssa.gov/oact/HistEst/CohLifeTables/{0}/CohLifeTables_{1}_Alt2_TR{0}.txt'
        columns = {'Year': 'year', 'x': 'age', 'l(x)': 'survivors'}
        for s in ('F', 'M'):
            response = self._session.get(url.format(Year.MAX_YEAR + 1, s))
            if not response.ok:
                return
            sleep(3)
            lines = [line.split() for line in response.text.splitlines()]
            df = pd.DataFrame(lines[6:], columns=lines[5])
            df = df[list(columns.keys())].rename(columns=columns)
            for col in df.columns:
                df[col] = df[col].map(int)
            df.to_csv(Filepath.ACTUARIAL.format(sex=s.lower()), index=False)


def main() -> None:
    refresher = SsaDataRefresher()
    refresher.refresh()
    if refresher.refreshed:
        create_all_generated_data()


if __name__ == '__main__':
    main()
