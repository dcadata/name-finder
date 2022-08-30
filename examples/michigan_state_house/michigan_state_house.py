import pandas as pd
import requests
from bs4 import BeautifulSoup
from scipy import stats

from prediction_getter_base import PredictionGetterBase


class StateRepsScraper:
    """
    Scrape current list of Michigan State Representatives
    """

    def scrape(self) -> None:
        self._scrape()
        self._clean()
        self._save()

    def _scrape(self) -> None:
        response = requests.get('https://www.house.mi.gov/AllRepresentatives')
        page = BeautifulSoup(response.text, 'lxml')
        representative_elems = page.find('ul', attrs=dict(id='allrepslist')).find_all('li')[1:]
        representative_data = [(
            rep.select_one('a.page-search-target'),
            rep.select_one('div.col-md-4'),
            rep.find('a', attrs=dict(href=lambda x: x.startswith('tel'))),
            rep.find('a', attrs=dict(href=lambda x: x.startswith('mailto'))),
        ) for rep in representative_elems]
        self._reps = pd.DataFrame(representative_data, columns=['rep', 'office', 'phone', 'email'])

    def _clean(self) -> None:
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

    def _save(self) -> None:
        self._reps.drop_duplicates().to_csv('representatives.csv', index=False)


class GenderPredictionGetter(PredictionGetterBase):
    def __init__(self):
        super().__init__(first_name_field='first_name', data_fp='representatives.csv')

    def get(self) -> None:
        self.get_predictions()
        self._predictions.to_csv('predictions.csv', index=False)
        self._data.merge(self._predictions, on=self._first_name_field).to_csv(
            'representatives_with_predictions.csv', index=False)


def summarize() -> None:
    data = pd.read_csv('representatives_with_predictions.csv')
    grouped_by_gender = data.groupby('prediction').first_name.count()
    grouped_by_gender_and_party = data.groupby(['party', 'prediction']).first_name.count()

    lines = [
        '# Gender Prediction Example - Michigan State Representatives',
        'Gender prediction compared to general population (assumed to be 50/50)',
    ]

    temp = tuple(grouped_by_gender[i] for i in ('M', 'F'))
    p_value = stats.chisquare(temp).pvalue
    p_value_status = '*' if p_value > 0.05 else ''
    lines.append('{}: Mx{}, Fx{} -> p={}{}'.format('All', *temp, round(p_value, 2), p_value_status))

    for major_party in ('Democrat', 'Republican'):
        temp = tuple(grouped_by_gender_and_party[major_party][i] for i in ('M', 'F'))
        p_value = stats.chisquare(temp).pvalue
        p_value_status = '*' if p_value > 0.05 else ''
        lines.append('{}: Mx{}, Fx{} -> p={}{}'.format(f'{major_party}s', *temp, round(p_value, 2), p_value_status))

    lines.append('*Not statistically significant')

    open('README.md', 'w').write('\n\n'.join(lines))


if __name__ == '__main__':
    summarize()
