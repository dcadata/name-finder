from time import sleep

import pandas as pd
import requests
from bs4 import BeautifulSoup
from scipy import stats

_API_BASE_URL = 'http://127.0.0.1:5000'


class Scraper:
    """
    Scrape current list of Michigan State Representatives
    """

    def scrape(self):
        self._scrape()
        self._clean()
        self._save()

    def _scrape(self):
        response = requests.get('https://www.house.mi.gov/AllRepresentatives')
        soup = BeautifulSoup(response.text, 'lxml')
        representative_elems = soup.find('ul', attrs=dict(id='allrepslist')).find_all('li')[1:]
        representative_data = [(
            rep.select_one('a.page-search-target'),
            rep.select_one('div.col-md-4'),
            rep.find('a', attrs=dict(href=lambda x: x.startswith('tel'))),
            rep.find('a', attrs=dict(href=lambda x: x.startswith('mailto'))),
        ) for rep in representative_elems]
        self._reps = pd.DataFrame(representative_data, columns=['rep', 'office', 'phone', 'email'])

    def _clean(self):
        self._reps = self._reps.dropna()

        self._reps.rep = self._reps.rep.apply(lambda x: x.text.strip())
        self._reps.office = self._reps.office.apply(lambda x: x.text.strip())
        self._reps.phone = self._reps.phone.apply(lambda x: x['href'].replace('tel:', '').strip())
        self._reps.email = self._reps.email.apply(lambda x: x['href'].replace('mailto:', '').strip())

        self._reps[['last_name', 'rep']] = self._reps.rep.str.split(', ', 1, expand=True)
        self._reps[['first_name', 'rep']] = self._reps.rep.str.split(' \(', 1, expand=True)
        self._reps[['party', 'rep']] = self._reps.rep.str.split('\) ', 1, expand=True)
        self._reps[['rep', 'district']] = self._reps.rep.str.split('-', 1, expand=True)
        self._reps = self._reps.drop(columns='rep')

    def _save(self):
        self._reps.to_csv('representatives.csv', index=False)


class GenderPredictor:
    """
    Predict genders of legislators based on first name
    """

    def predict(self):
        self._read_scraped_data()
        self._get_gender_predictions()
        self._save()

    def _read_scraped_data(self):
        self._reps = pd.read_csv('representatives.csv').drop_duplicates()

    def _get_gender_predictions(self):
        self._predictions = []
        session = requests.Session()
        for name in self._reps.first_name.unique():
            if response := session.get(f'{_API_BASE_URL}/predict/gender/{name}', params=dict(
                    before=2001, living=1)).json():  # age limit of 21+
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
        self._reps.merge(predictions, on='first_name').to_csv('representatives_with_predictions.csv', index=False)


def summarize():
    df = pd.read_csv('representatives_with_predictions.csv').dropna()
    df = df[(df.gender_confidence >= 0.8) & (df.gender_number >= 25)].copy()  # drop low-confidence predictions
    grouped_by_gender = df.groupby('gender').first_name.count()
    grouped_by_gender_and_party = df.groupby(['party', 'gender']).first_name.count()

    lines = [
        '# Gender Prediction Example - Michigan State Representatives',
        'Gender prediction compared to general population (assumed to be 50/50)',
    ]

    data = tuple(grouped_by_gender[i] for i in ('M', 'F'))
    p_value = stats.chisquare(data).pvalue
    p_value_status = '*' if p_value > 0.05 else ''
    lines.append('{}: Mx{}, Fx{} -> p={}{}'.format('All', *data, round(p_value, 2), p_value_status))

    for major_party in ('Democrat', 'Republican'):
        data = tuple(grouped_by_gender_and_party[major_party][i] for i in ('M', 'F'))
        p_value = stats.chisquare(data).pvalue
        p_value_status = '*' if p_value > 0.05 else ''
        lines.append('{}: Mx{}, Fx{} -> p={}{}'.format(f'{major_party}s', *data, round(p_value, 2), p_value_status))

    lines.append('*Not statistically significant')

    open('README.md', 'w').write('\n\n'.join(lines))


if __name__ == '__main__':
    summarize()
