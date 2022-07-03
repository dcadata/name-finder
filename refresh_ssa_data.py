import zipfile
from time import sleep

import pandas as pd
import requests

from finder import MAX_YEAR


def _refresh_babynames(session: requests.Session) -> bool:
    # compare to website
    response = session.get('https://www.ssa.gov/oact/babynames/limits.html')
    table = pd.read_html(response.text)[0]
    if MAX_YEAR >= int(table.iloc[0, 0]):
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


def _refresh_actuarial(session: requests.Session) -> None:
    url = 'https://www.ssa.gov/oact/HistEst/CohLifeTables/{0}/CohLifeTables_{1}_Alt2_TR{0}.txt'
    columns = {'Year': 'year', 'x': 'age', 'l(x)': 'survivors'}
    for s in ('F', 'M'):
        response = session.get(url.format(MAX_YEAR + 1, s))
        if not response.ok:
            return
        sleep(3)
        lines = [line.split() for line in response.text.splitlines()]
        df = pd.DataFrame(lines[6:], columns=lines[5])
        df = df[list(columns.keys())].rename(columns=columns)
        for col in df.columns:
            df[col] = df[col].apply(int)
        df.to_csv(f'data/actuarial/{s.lower()}.csv', index=False)


def main() -> None:
    session = requests.Session()
    if _refresh_babynames(session):
        _refresh_actuarial(session)
    session.close()


if __name__ == '__main__':
    main()