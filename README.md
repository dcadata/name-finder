# Neuname Name API

## Feature Overview

AGE PREDICTION

age distribution of people with a given first name
returns a condensed probability distribution of ages
optionally, can take gender into account
competing products generally do not return a condensed probability distribution of ages (they only return a single mean age), nor take gender into account when predicting age

GENDER PREDICTION

gender of people with a given first name
returns gender prediction with confidence
optionally, can take age into account--this does not matter for most names, but matters for some traditionally masculine names that have trended feminine (e.g. Leslie)
competing products generally do not take age into account when predicting gender

SEARCH

search for first names based on their characteristics

INFO

get info and trends/history for a name

## API Documentation

x

## Examples

### Example usages of `predict_age`

Predict age of a person named Dorothy

    >>> self.predict_age('dorothy')
    {'name': 'Dorothy', 'sex': None, 'exclude_deceased': False, 'buckets': None, 'prediction': {'mean': 89, 'percentiles': {0.68: {'min': 81, 'max': 106}, 0.95: {'min': 57, 'max': 115}, 0.997: {'min': 0, 'max': 119}}}}

Predict age of a person named Dorothy, excluding deceased by probability (via SSA actuarial tables)

    >>> self.predict_age('dorothy', exclude_deceased=True)
    {'name': 'Dorothy', 'sex': None, 'exclude_deceased': True, 'buckets': None, 'prediction': {'mean': 78, 'percentiles': {0.68: {'min': 70, 'max': 96}, 0.95: {'min': 46, 'max': 105}, 0.997: {'min': 0, 'max': 107}}}}

Predict ages of male and female Leslies; note difference in mean age and distributions

> Note: passing `sex` won't make much difference unless the name has trended masculine or feminine over time. The majority of names have no such trend (e.g. Jessica, Matthew, etc.)

    >>> self.predict_age('leslie', 'm')
    {'name': 'Leslie', 'sex': 'M', 'exclude_deceased': False, 'buckets': None, 'prediction': {'mean': 74, 'percentiles': {0.68: {'min': 55, 'max': 105}, 0.95: {'min': 30, 'max': 110}, 0.997: {'min': 5, 'max': 119}}}}

    >>> self.predict_age('leslie', 'f')
    {'name': 'Leslie', 'sex': 'F', 'exclude_deceased': False, 'buckets': None, 'prediction': {'mean': 45, 'percentiles': {0.68: {'min': 16, 'max': 68}, 0.95: {'min': 8, 'max': 75}, 0.997: {'min': 0, 'max': 104}}}}

Use `buckets=4` to indicate quartiles

    >>> self.predict_age('catherine', buckets=4, exclude_deceased=True)
    {'name': 'Catherine', 'sex': None, 'exclude_deceased': True, 'buckets': 4, 'prediction': {'mean': 54, 'percentiles': {0.25: {'min': 59, 'max': 69}, 0.5: {'min': 29, 'max': 74}, 0.75: {'min': 19, 'max': 79}, 1.0: {'min': 0, 'max': 119}}}}

Practical example: 95% of Taylors are <= 34. 95% of Aidens are <= 16. This has implications for what kind of advertising, etc., could be most relevant to a customer named Aiden whose age you don't know.

    >>> self.predict_age('taylor')
    {'name': 'Taylor', 'sex': None, 'exclude_deceased': False, 'buckets': None, 'prediction': {'mean': 22, 'percentiles': {0.68: {'min': 17, 'max': 30}, 0.95: {'min': 2, 'max': 34}, 0.997: {'min': 0, 'max': 106}}}}
    >>> self.predict_age('aiden')
    {'name': 'Aiden', 'sex': None, 'exclude_deceased': False, 'buckets': None, 'prediction': {'mean': 9, 'percentiles': {0.68: {'min': 4, 'max': 13}, 0.95: {'min': 0, 'max': 16}, 0.997: {'min': 0, 'max': 23}}}}

For some names--particularly those that have been in use for generations, but are trending up in recent years--excluding deceased individuals shows a meaningful difference. Without excluding deceased individuals, 95% of Avas are <= 67. When excluding deceased individuals, 95% of Avas are <= 20.

    >>> self.predict_age('ava')
    {'name': 'Ava', 'sex': None, 'exclude_deceased': False, 'buckets': None, 'prediction': {'mean': 12, 'percentiles': {0.68: {'min': 3, 'max': 14}, 0.95: {'min': 0, 'max': 67}, 0.997: {'min': 0, 'max': 108}}}}
    >>> self.predict_age('ava', exclude_deceased=True)
    {'name': 'Ava', 'sex': None, 'exclude_deceased': True, 'buckets': None, 'prediction': {'mean': 11, 'percentiles': {0.68: {'min': 3, 'max': 14}, 0.95: {'min': 0, 'max': 20}, 0.997: {'min': 0, 'max': 89}}}}

### Example usages of `predict_gender`

Predict gender of Leslies born in 1930 vs 2000

    >>> self.predict_gender('leslie', 1930)
    {'name': 'Leslie', 'birth_year_range': [1928, 1929, 1930, 1931, 1932], 'exclude_deceased': False, 'prediction': 'M', 'confidence': 0.92}
    >>> self.predict_gender('leslie', 2000)
    {'name': 'Leslie', 'birth_year_range': [1998, 1999, 2000, 2001, 2002], 'exclude_deceased': False, 'prediction': 'F', 'confidence': 0.97}

Predict gender of Marions born in 1920 vs 2010

    >>> self.predict_gender('marion', 1920)
    {'name': 'Marion', 'birth_year_range': [1918, 1919, 1920, 1921, 1922], 'exclude_deceased': False, 'prediction': 'F', 'confidence': 0.78}
    >>> self.predict_gender('marion', 2010)
    {'name': 'Marion', 'birth_year_range': [2008, 2009, 2010, 2011, 2012], 'exclude_deceased': False, 'prediction': 'M', 'confidence': 0.53}

However, you don't *have* to specify birth year--in this case, all birth years will be included

    >>> self.predict_gender('elizabeth')
    {'name': 'Elizabeth', 'birth_year_range': [], 'exclude_deceased': False, 'prediction': 'F', 'confidence': 1.0}
    >>> self.predict_gender('casey')
    {'name': 'Casey', 'birth_year_range': [], 'exclude_deceased': False, 'prediction': 'M', 'confidence': 0.59}
    >>> self.predict_gender('george')
    {'name': 'George', 'birth_year_range': [], 'exclude_deceased': False, 'prediction': 'M', 'confidence': 0.99}

## Data Sources

This project uses United States Social Security Administration (SSA) data available via ["Beyond the Top 1000 Names" at SSA.gov](https://www.ssa.gov/oact/babynames/limits.html). National data was combined with territory-specific data. 

Actuarial tables [also via SSA](https://www.ssa.gov/oact/HistEst/CohLifeTablesHome.html).

## Caveats

[Some important background and limitations, per SSA:](https://www.ssa.gov/oact/babynames/background.html)

>- All names are from Social Security card applications for births that occurred in the United States after 1879. Note that many people born before 1937 never applied for a Social Security card, so their names are not included in our data. For others who did apply, our records may not show the place of birth, and again their names are not included in our data.
>- Names are restricted to cases where the year of birth, sex, and state of birth are on record, and where the given name is at least 2 characters long.
>- Name data are tabulated from the "First Name" field of the Social Security Card Application. Hyphens and spaces are removed, thus Julie-Anne, Julie Anne, and Julieanne will be counted as a single entry.
>- Name data are not edited. For example, the sex associated with a name may be incorrect. Entries such as "Unknown" and "Baby" are not removed from the lists.
>- To safeguard privacy, we exclude from our tabulated lists of names those that would indicate, or would allow the ability to determine, names with fewer than 5 occurrences in any geographic area. If a name has less than 5 occurrences for a year of birth in any state, the sum of the state counts for that year will be less than the national count.

## Future Features

- combine with Google trends to correlate rises/falls in popularity of name w/ trends around that name
- game in which users guess (without looking it up) which of two names is more common
