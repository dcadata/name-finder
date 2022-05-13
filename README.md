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

### SEARCH BY TEXT

* Allows search functionality via natural language string

### NAME

* Get info and history for specified first name

### COMPARE

* Compare two or more names

## API Documentation & Examples

### Using `predict_age`

#### Params

* `name` - string, required: the name for which you want a prediction
* `gender` - string, optional: gender of individual - pass "m" or "f" if available; defaults to none
* `living` - integer, optional: pass 1 if you want to include only living individuals; defaults to 0
* `buckets` - integer, optional: number of buckets in probability distribution, e.g. pass 5 for quintiles; defaults to 0.68-0.95-0.997

#### Examples

Predict ages of male and female Lindseys

    predict/age/lindsey?gender=m
    predict/age/lindsey?gender=f

Note: passing gender won't make a difference for most names, but if you know the gender, you can get a more accurate prediction by passing it.

Using buckets: e.g. use `buckets=4` to indicate quartiles

    predict/age/marcus?buckets=4

Including only living individuals can show a meaningful difference for some names. When including both living and deceased individuals, 95% of Avas are <= age 68 (as of 2021 data). When including only living individuals, 95% of Avas are <= age 21 (as of 2021 data).

    predict/age/ava
    predict/age/ava?living=1

Practical example: 95% of Taylors are between age 3 and age 36 (as of 2021 data). This has implications for what kind of advertising, etc., could be most relevant to a customer named Taylor whose age isn't otherwise known.

    predict/age/taylor

### Using `predict_gender`

#### Params

* `name` - string, required: the name for which you want a prediction
* `year` - integer, optional: year in which individual was born
* `living` - integer, optional: pass 1 if you want to include only living individuals; defaults to 0

#### Examples

For most names, gender can be predicted with (near-)certainty:

    predict/gender/barbara
    predict/gender/carlos
    predict/gender/dante
    predict/gender/ellen

However, this is not true of all names:

    predict/gender/jordan
    predict/gender/krishna

If you know the birth year, passing it can allow a more accurate gender prediction. Otherwise, all years will be included.

Predict gender of Leslies born in 1940, 1980, and 2000

    predict/gender/leslie?year=1940
    predict/gender/leslie?year=1980
    predict/gender/leslie?year=2000

### Using `search`

#### Params

* coming soon

#### Examples

Names that both start and end with "A", and are 3-5 letters long

    search?start=a&end=a&length=3,5

Masculine names ending in "A" (and similar sounds) that aren't super rare

    search?gender=0.8,1&end=a,ah,ay,ai,ae&number_min=1000

Feminine names starting with "E" or "I" that have gained at least 10% in popularity since 2010

    search?gender=0,0.2&start=e,i&delta_after=2010&delta_pct=0.1

Short names that were neutral before 1990 and have trended at least 1% less popular and 1% more masculine since 1990

    search?length=3,5&gender=0.2,0.8&before=1990&delta_after=1990&delta_pct=-0.01&delta_masc=0.01

Variations of a name, using regex pattern

    search?pattern=^e?[ck]ath?e?r[iy]nn?[ea]?$  # Catherine
    search?pattern=^v[iy][ck]{1,2}tor[iye]{1,2}a$  # Victoria
    search?pattern=^ja[yie]?d[eiyao]n$  # Ja(y)den
    search?pattern=^gabriell?[ea]?$  # variations of Gabriel

### Using `name`

#### Params

* `name` - string, required: the name for which you want information
* `after` - integer, optional: year of birth after which to filter (inclusive)
* `before` - integer, optional: year of birth before which to filter (inclusive)
* `show_historic` - integer, optional: pass 1 if you want to show historic data; defaults to 0

#### Examples

    name/darren
    name/helen?after=2010
    name/zachary?after=1980&before=2000

### Using `compare`

#### Params

* `names` - hyphen-separated strings, required: the names for which you want information
* remaining params same as `name` endpoint

#### Examples

    compare/jeremy-jeremiah
    compare/michelle-mikayla-michaela?before=2000
    compare/esmeralda-emerald?after=1950
    compare/leander-leandra?after=1915&before=2015
