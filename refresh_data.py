import zipfile
from io import StringIO
from time import sleep

import pandas as pd
from requests import Session

from core import Filepath, build_all_generated_data


class SsaDataDownloader:
    def __init__(self, max_year: int) -> None:
        self._max_year: int = max_year

    def download(self) -> None:
        self._open_session()
        self._download_name_data()
        self._download_applicants_data()
        self._download_actuarial_data()
        self._close_session()
        return

    def _open_session(self) -> None:
        self._session: Session = Session()
        self._session.headers.update({'User-Agent': f'name-finder/{self._max_year - 1} Update'})
        return

    def _close_session(self) -> None:
        self._session.close()
        return

    def _download_name_data(self) -> None:
        for url in (
                'https://www.ssa.gov/oact/babynames/names.zip',
                'https://www.ssa.gov/oact/babynames/state/namesbystate.zip',
                'https://www.ssa.gov/oact/babynames/territory/namesbyterritory.zip',
        ):
            filepath = Filepath.DATA_DIR + url.rsplit('/', 1)[1]
            with open(filepath, 'wb') as f:
                response = self._session.get(url)
                for chunk in response.iter_content(chunk_size=128):
                    f.write(chunk)
            sleep(3)
            with zipfile.ZipFile(filepath) as z:
                z.extractall(filepath[:-4])
        return

    def _download_applicants_data(self) -> None:
        response = self._session.get('https://www.ssa.gov/oact/babynames/numberUSbirths.html')
        html = StringIO(response.text)
        table = pd.read_html(html)[0]
        table = table.rename(columns=dict((col, ''.join(col.split())) for col in table.columns)).rename(columns=dict(
            Yearofbirth='year', Male='number_m', Female='number_f', Total='number'))
        table.to_csv(Filepath.APPLICANTS_DATA, index=False)
        return

    def _download_actuarial_data(self) -> None:
        url = 'https://www.ssa.gov/oact/HistEst/CohLifeTables/{0}/CohLifeTables_{1}_Alt2_TR{0}.txt'
        columns = {'Year': 'year', 'x': 'age', 'l(x)': 'survivors'}
        for s in ('F', 'M'):
            response = self._session.get(url.format(self._max_year + 1, s))
            if not response.ok:
                return
            sleep(3)
            lines = [line.split() for line in response.text.splitlines()]
            df = pd.DataFrame(lines[6:], columns=lines[5])
            df = df[list(columns.keys())].rename(columns=columns)
            for col in df.columns:
                df[col] = df[col].map(int)
            df.to_csv(Filepath.ACTUARIAL.format(sex=s.lower()), index=False)
        return


def main() -> None:
    downloader = SsaDataDownloader(2025)
    downloader.download()
    build_all_generated_data()
    return


if __name__ == '__main__':
    main()
