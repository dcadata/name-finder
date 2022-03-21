import zipfile
from time import sleep

import pandas as pd
import requests

from finder import _MAX_YEAR


def _run(session):
    # compare to website
    response = session.get('https://www.ssa.gov/oact/babynames/limits.html')
    table = pd.read_html(response.text)[0]
    if _MAX_YEAR >= int(table.iloc[0, 0]):
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
