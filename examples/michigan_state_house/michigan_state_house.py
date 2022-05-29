from time import sleep

import pandas as pd
import requests
from bs4 import BeautifulSoup
from scipy import stats

_API_BASE_URL = 'http://127.0.0.1:5000'


class Scraper:
    def scrape(self):
        self._scrape()
        self._clean()
        self._save()

    def _scrape(self):
        response = requests.get('https://www.house.mi.gov/AllRepresentatives?handler=SortReps&sortby=Alpha')
        soup = BeautifulSoup(response.text, 'lxml')
        representative_elems = soup.select('li')
        representative_data = [(
            rep.select_one('a.page-search-target'),
            rep.select_one('div.col-md-4'),
            rep.find('a', attrs=dict(href=lambda x: x.startswith('tel'))),
            rep.find('a', attrs=dict(href=lambda x: x.startswith('mailto'))),
        ) for rep in representative_elems]
        self._data = pd.DataFrame(representative_data, columns=['rep', 'office', 'phone', 'email'])

    def _clean(self):
        self._data = self._data.dropna()

        self._data.rep = self._data.rep.apply(lambda x: x.text.strip())
        self._data.office = self._data.office.apply(lambda x: x.text.strip())
        self._data.phone = self._data.phone.apply(lambda x: x['href'].replace('tel:', '').strip())
        self._data.email = self._data.email.apply(lambda x: x['href'].replace('mailto:', '').strip())

        self._data[['last_name', 'rep']] = self._data.rep.str.split(', ', 1, expand=True)
        self._data[['first_name', 'rep']] = self._data.rep.str.split(' \(', 1, expand=True)
        self._data[['party', 'rep']] = self._data.rep.str.split('\) ', 1, expand=True)
        self._data[['rep', 'district']] = self._data.rep.str.split('-', 1, expand=True)
        self._data = self._data.drop(columns='rep')

    def _save(self):
        self._data.to_csv('representatives.csv', index=False)


class Predictor:
    def predict(self):
        self._read_scraped_data()
        self._get_gender_predictions()
        self._save()

    def _read_scraped_data(self):
        self._data = pd.read_csv('representatives.csv').drop_duplicates()

    def _get_gender_predictions(self):
        self._predictions = []
        session = requests.Session()
        for name in self._data.first_name.unique():
            response = session.get(f'{_API_BASE_URL}/predict/gender/{name}', params=dict(
                before=2001, living=1)).json()  # age limit of 21+
            if not response:
                continue
            self._predictions.append(dict(
                first_name=name,
                gender=response['prediction'],
                gender_confidence=response['confidence'],
                gender_number=response['number'],
            ))
            sleep(1)
        session.close()

    def _save(self):
        predictions = pd.DataFrame(self._predictions)
        predictions.to_csv('predictions.csv', index=False)

        data = self._data.merge(predictions, on='first_name')
        data.to_csv('representatives_with_predictions.csv', index=False)


def summarize():
    df = pd.read_csv('representatives_with_predictions.csv').dropna()
    df = df[(df.gender_confidence >= 0.8) & (df.gender_number >= 25)].copy()  # drop low-confidence predictions
    grouped_by_gender = df.groupby(['party', 'gender']).first_name.count()

    output = ['GENDER - compared to general population']
    for major_party in ('Democrat', 'Republican'):
        data = (grouped_by_gender[major_party]['F'], grouped_by_gender[major_party]['M'])
        p_value = stats.chisquare(data).pvalue
        p_value_status = '*' if p_value > 0.05 else ''
        output.append('{}: Fx{}, Mx{} -> p={}{}'.format(major_party[0], *data, round(p_value, 2), p_value_status))

    print('\n'.join(output))


if __name__ == '__main__':
    summarize()
