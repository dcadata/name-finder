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


def _refresh_actuarial_tables(session):
    response = session.get('https://www.ssa.gov/oact/HistEst/Death/2021/DeathProbsE_M_Hist_TR2021.txt')
    lines = [line.split() for line in response.text.splitlines()[1:]]
    columns = lines[0]
    table = pd.DataFrame(lines[1:], columns=columns)
    table.to_csv('data/actuarial.csv', index=False)


def main():
    session = requests.Session()
    if _refresh_babynames(session):
        _refresh_actuarial_tables(session)
    session.close()


if __name__ == '__main__':
    main()
