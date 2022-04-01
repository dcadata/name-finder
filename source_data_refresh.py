import zipfile
from time import sleep

import pandas as pd
import requests

from finder import _MAX_YEAR


def _refresh_babynames(session):
    # compare to website
    response = session.get('https://www.ssa.gov/oact/babynames/limits.html')
    table = pd.read_html(response.text)[0]
    if _MAX_YEAR >= int(table.iloc[0, 0]):
        return False

    sleep(3)
    # if year has been added, then download files
    for url in (
            'https://www.ssa.gov/oact/babynames/names.zip',
            'https://www.ssa.gov/oact/babynames/territory/namesbyterritory.zip',
    ):
        filepath = 'data/' + url.rsplit('/', 1)[1]
        open(filepath, 'wb').write(session.get(url).content)
        sleep(3)
        with zipfile.ZipFile(filepath) as z:
            z.extractall(filepath[:-4])

    return True


def _refresh_actuarial(session):
    max_year = _MAX_YEAR + 2
    for s in ('F', 'M'):
        url = f'https://www.ssa.gov/oact/HistEst/CohLifeTables/{max_year}/CohLifeTables_{s}_Alt2_TR{max_year}.txt'
        response = session.get(url)
        sleep(3)
        lines = [line.split() for line in response.text.splitlines()]
        table = pd.DataFrame(lines[6:], columns=lines[5])
        columns = {'Year': 'year', 'x': 'age', 'l(x)': 'survivors'}
        table = table[list(columns.keys())].rename(columns=columns)
        for col in table.columns:
            table[col] = table[col].apply(int)
        table.to_csv(f'data/actuarial/{s.lower()}.csv', index=False)


def main():
    session = requests.Session()
    if _refresh_babynames(session):
        _refresh_actuarial(session)
    session.close()


if __name__ == '__main__':
    main()
