import zipfile
from time import sleep

import requests


def main():
    urls = (
        'https://www.ssa.gov/oact/babynames/names.zip',
        'https://www.ssa.gov/oact/babynames/territory/namesbyterritory.zip',
    )

    session = requests.Session()

    for url in urls:
        filepath = 'data/' + url.rsplit('/', 1)[1]
        open(filepath, 'wb').write(session.get(url).content)
        sleep(3)
        with zipfile.ZipFile(filepath) as z:
            z.extractall(filepath[:-4])

    session.close()


if __name__ == '__main__':
    main()
