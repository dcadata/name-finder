import zipfile
from datetime import datetime
from time import sleep

import pandas as pd
import requests


def _run(session):
    # read table on page to see if year has been added
    response = session.get('https://www.ssa.gov/oact/babynames/limits.html')
    df = pd.read_html(response.text)[0]
    if df.iloc[0, 0] != datetime.today().year - 1:
        return

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


def main():
    session = requests.Session()
    _run(session)
    session.close()


if __name__ == '__main__':
    main()
