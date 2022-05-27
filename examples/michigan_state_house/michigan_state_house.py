from time import sleep

import pandas as pd
import requests
from bs4 import BeautifulSoup
from scipy import stats

_REPRESENTATIVES_DATA_PATH = 'representatives.csv'
_API_BASE_URL = 'http://127.0.0.1:5000'


class Scraper:
    def run(self):
        self._scrape_representative_info()
        self._arrange_representative_info()
        self._clean_representative_info()
        self._get_gender_predictions()
        self._save()

    def _scrape_representative_info(self):
        response = requests.get('https://www.house.mi.gov/AllRepresentatives?handler=SortReps&sortby=Alpha')
        soup = BeautifulSoup(response.text, 'lxml')
        self._representative_elems = soup.select('li')

    def _arrange_representative_info(self):
        representative_data = [(
            rep.select_one('a.page-search-target'),
            rep.select_one('div.col-md-4'),
            rep.find('a', attrs=dict(href=lambda x: x.startswith('tel'))),
            rep.find('a', attrs=dict(href=lambda x: x.startswith('mailto'))),
        ) for rep in self._representative_elems]
        self.data = pd.DataFrame(representative_data, columns=['rep', 'office', 'phone', 'email'])

    def _clean_representative_info(self):
        self.data = self.data.dropna()

        self.data.rep = self.data.rep.apply(lambda x: x.text.strip())
        self.data.office = self.data.office.apply(lambda x: x.text.strip())
        self.data.phone = self.data.phone.apply(lambda x: x['href'].replace('tel:', '').strip())
        self.data.email = self.data.email.apply(lambda x: x['href'].replace('mailto:', '').strip())

        self.data[['last_name', 'rep']] = self.data.rep.str.split(', ', 1, expand=True)
        self.data[['first_name', 'rep']] = self.data.rep.str.split(' \(', 1, expand=True)
        self.data[['party', 'rep']] = self.data.rep.str.split('\) ', 1, expand=True)
        self.data[['rep', 'district']] = self.data.rep.str.split('-', 1, expand=True)
        self.data = self.data.drop(columns='rep')

    def _get_gender_predictions(self):
        prediction = []
        confidence = []

        session = requests.Session()
        for name in self.data.first_name:
            try:
                response = session.get(f'{_API_BASE_URL}/predict/gender/{name}?living=1').json()
                prediction.append(response['prediction'])
                confidence.append(response['confidence'])
            except KeyError:
                prediction.append(None)
                confidence.append(None)
            sleep(1)
        session.close()

        self.data = self.data.assign(gender=prediction).assign(gender_confidence=confidence)

    def _save(self):
        self.data.to_csv(_REPRESENTATIVES_DATA_PATH, index=False)


def summarize():
    df = pd.read_csv(_REPRESENTATIVES_DATA_PATH).dropna()
    df = df[df.gender_confidence >= 0.8].copy()  # drop low-confidence predictions
    grouped_by_gender = df.groupby(['party', 'gender']).first_name.count()

    output = ['GENDER - compared to general population']
    for major_party in ('Democrat', 'Republican'):
        data = (grouped_by_gender[major_party]['F'], grouped_by_gender[major_party]['M'])
        # noinspection PyUnresolvedReferences
        p_value = stats.chisquare(data).pvalue
        p_value_status = '*' if p_value > 0.05 else ''
        output.append('{}: Fx{}, Mx{} -> p={}{}'.format(major_party[0], *data, round(p_value, 2), p_value_status))

    print('\n'.join(output))


if __name__ == '__main__':
    summarize()
