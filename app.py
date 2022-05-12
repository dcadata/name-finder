import re

from flask import Flask, request, jsonify
from markupsafe import escape

import finder

finder._OUTPUT_RECORDS = True

displayer = finder.Displayer()
displayer.load()

app = Flask(__name__)


def _escape_optional_string(arg_name: str):
    value = request.args.get(arg_name, default=None, type=str)
    return escape(value) if value else None


def _escape_optional_string_into_list(arg_name: str):
    value = request.args.get(arg_name, default=None, type=str)
    if not value:
        return None
    return escape(value).split(',')


def _escape_optional_string_into_list_of_floats(arg_name: str):
    values = _escape_optional_string_into_list(arg_name)
    if not values:
        return
    return list(map(float, values))


def _escape_optional_string_into_list_of_ints(arg_name: str):
    values = _escape_optional_string_into_list(arg_name)
    if not values:
        return
    return list(map(int, values))


def _name_base(name1: str) -> dict:
    data = displayer.name(
        name=escape(name1),
        after=request.args.get('after', default=None, type=int),
        before=request.args.get('before', default=None, type=int),
    )
    return data


@app.route('/name/<string:name1>')
def name_endpoint(name1: str):
    return jsonify(_name_base(name1))


@app.route('/compare')
def compare_endpoint():
    names = _escape_optional_string_into_list('names')
    data = [_name_base(name) for name in names]
    return jsonify(data)


@app.route('/search')
def search_endpoint():
    data = displayer.search(
        pattern=_escape_optional_string('pattern'),
        start=_escape_optional_string_into_list('start'),
        end=_escape_optional_string_into_list('end'),
        contains=_escape_optional_string_into_list('contains'),
        contains_any=_escape_optional_string_into_list('contains_any'),
        not_start=_escape_optional_string_into_list('not_start'),
        not_end=_escape_optional_string_into_list('not_end'),
        not_contains=_escape_optional_string_into_list('not_contains'),
        order=_escape_optional_string_into_list('order'),
        length=_escape_optional_string_into_list_of_ints('length'),
        number_min=request.args.get('number_min', default=None, type=int),
        number_max=request.args.get('number_max', default=None, type=int),
        gender=_escape_optional_string_into_list_of_floats('gender'),
        after=request.args.get('after', default=None, type=int),
        before=request.args.get('before', default=None, type=int),
        delta_after=request.args.get('delta_after', default=None, type=int),
        delta_pct=request.args.get('delta_pct', default=None, type=float),
        delta_fem=request.args.get('delta_fem', default=None, type=float),
        delta_masc=request.args.get('delta_masc', default=None, type=float),
    )
    return jsonify(data)


@app.route('/search_by_text')
def search_by_text_endpoint():
    text = request.args.get('text', default=None, type=str)
    if not text:
        return {}
    text = escape(text.lower())
    delta_sections = re.split('trend(ing|ed)?\s', text, 1)
    if len(delta_sections) > 1:
        text, delta_section = delta_sections[0], delta_sections[-1]
    else:
        delta_section = ''

    def _safely_check_regex(pattern: str):
        try:
            return re.search(pattern, text).groups()[-1]
        except (AttributeError, IndexError):
            return

    def _safely_check_regex_and_split_into_tuple(pattern: str):
        result = _safely_check_regex(pattern)
        return tuple(result.split(',')) if result else None

    def _safely_check_regex_delta_section(pattern: str):
        try:
            return re.search(pattern, delta_section).groups()[-1]
        except (AttributeError, IndexError):
            return

    length_ind = _safely_check_regex('(short|med|long)')
    gender_ind = _safely_check_regex('(fem|neu|unisex|masc)')
    after_ind = _safely_check_regex('(after|since)\s([0-9]{4})')
    before_ind = _safely_check_regex('before\s([0-9]{4})')
    delta_after_ind = _safely_check_regex_delta_section('(after|since)\s([0-9]{4})')
    delta_pct_ind = _safely_check_regex_delta_section('(down|up)')
    delta_gender_ind = _safely_check_regex_delta_section('(fem|masc)')

    delta_after = int(delta_after_ind) if delta_after_ind else None
    if not delta_after and (delta_pct_ind or delta_gender_ind):  # suggests they intended to add a trend
        delta_after = finder.MAX_YEAR - 20
    delta_pct = dict(down=-0.001, up=0.001).get(delta_pct_ind)
    delta_fem = dict(fem=0.001, masc=-0.001).get(delta_gender_ind)

    conditions = dict(
        pattern=_safely_check_regex('(pattern|regex)\s(.*)'),
        start=_safely_check_regex_and_split_into_tuple('(start|beginn?)(ing|s)?(\swith)?\s([a-z,]+)'),
        end=_safely_check_regex_and_split_into_tuple('end(ing|s)?(\swith)?\s([a-z,]+)'),
        contains=_safely_check_regex_and_split_into_tuple('contain(ing|s)?\s([a-z,]+)'),
        order=_safely_check_regex_and_split_into_tuple('order\s([a-z,]+)'),
        length=dict(short=(3, 5), med=(6, 8), long=(9, 30)).get(length_ind),
        gender=dict(fem=(0, 0.2), neu=(0.2, 0.8), unisex=(0.2, 0.8), masc=(0.8, 1)).get(gender_ind),
        after=int(after_ind) if after_ind else None,
        before=int(before_ind) if before_ind else None,
        delta_after=delta_after,
        delta_pct=delta_pct,
        delta_fem=delta_fem,
    )
    data = displayer.search(**conditions)
    if data:
        data = data[:30]
    data = dict(conditions=conditions, bot_text=', '.join('{name}({display})'.format(**i) for i in data), data=data)
    return jsonify(data)


@app.route('/predict/age/<string:name1>')
def predict_age_endpoint(name1: str):
    data = displayer.predict_age(
        name=escape(name1),
        gender=_escape_optional_string('gender'),
        living=bool(request.args.get('living', default=0, type=int)),
        buckets=request.args.get('buckets', default=None, type=int),
    )
    return jsonify(data)


@app.route('/predict/gender/<string:name1>')
def predict_gender_endpoint(name1: str):
    data = displayer.predict_gender(
        name=escape(name1),
        birth_year=request.args.get('birth_year', default=None, type=int),
        living=bool(request.args.get('living', default=0, type=int)),
    )
    return jsonify(data)


if __name__ == '__main__':
    app.run()
