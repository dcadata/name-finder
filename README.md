# NameFirst name-based prediction & name search tool

## Feature Overview

### AGE PREDICTION

* Returns age distribution of people with a specified first name
* Optionally, can take gender into account
* Similar tools generally do not return a probability distribution of ages (they only return a single mean age), nor take gender into account when predicting age

### GENDER PREDICTION

* Returns gender of people with a specified first name (along with confidence level in the prediction)
* Optionally, can take age into account
* Competing products generally do not take age into account when predicting gender

### SEARCH

* Search for first names based on their characteristics

### NAME

* Get info and history for specified first name

### COMPARE

* Compare two or more names

## API Documentation

(coming eventually)

## Examples

### Using `predict_age`

Predict ages of male and female Leslies

    predict/age/leslie?gender=m
    predict/age/leslie?gender=f

Note: Passing gender won't make a difference for most names, but if you know the gender, you can get a more accurate prediction by passing it.

Using buckets: e.g. use `buckets=4` to indicate quartiles

    predict/age/dorothy?buckets=4

For some names--particularly those that have been in use for generations, but are trending up in recent years--excluding deceased individuals shows a meaningful difference. Without excluding deceased individuals, 95% of Avas are <= 67. When excluding deceased individuals, 95% of Avas are <= 20.

    predict/age/ava
    predict/age/ava?exclude_deceased=true

Practical example: 95% of Taylors are <= 34. 95% of Aidens are <= 16. This has implications for what kind of advertising, etc., could be most relevant to a customer named Aiden whose age you don't know.

    predict/age/taylor
    predict/age/aiden

### Using `predict_gender`

For most names, gender can be predicted with relative certainty:

    predict/gender/jessica
    predict/gender/michael

If you know the birth year, passing it can allow a more confident gender prediction. Otherwise, all years will be included.

Predict gender of Leslies born in 1940, 1980, and 2000

    predict/gender/leslie?year=1940
    predict/gender/leslie?year=1980
    predict/gender/leslie?year=2000

Predict gender of Marions born in 1920 and 2020

    predict/gender/marion?year=1920
    predict/gender/marion?year=2020

### Using `search`

Names that both start and end with "A", and are 3 letters long

    search?start=a&end=a&length=3,3

Masculine names ending in "A" (and similar sounds) that aren't super rare

    search?gender=0.9,1&end=a,ah,ay,ai,ae&number_min=1000

Feminine names starting with "E" or "I" that have gained at least 10% in popularity since 2010

    search?gender=0,0.1&start=e,i&delta_after=2010&delta_pct=0.1

Short names that were neutral before 1990 and have trended at least 1% less popular and 1% more masculine since 1990

    search?length=3,5&gender=0.2,0.8&before=1990&delta_after=1990&delta_pct=-0.01&delta_masc=0.01

Variations of a name, using regex pattern

    search?pattern=^e?[ck]ath?e?r[iy]nn?[ea]?$  # Catherine
    search?pattern=^v[iy][ck]{1,2}tor[iye]{1,2}a$  # Victoria
    search?pattern=^ja[yie]?d[eiyao]n$  # Ja(y)den

Feminine variations of the name Cory

    search?gender=0,0.2&pattern=^[ck]orr?(e?y|ie?|ii|ee)$  # doesn't include variations of Corinne/a

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
