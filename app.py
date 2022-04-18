from flask import Flask, request, jsonify
from markupsafe import escape

import finder

finder._OUTPUT_RECORDS = True

displayer = finder.Displayer()
displayer.load()

app = Flask(__name__)


def _escape_optional_string(arg_name):
    value = request.args.get(arg_name, default=None, type=str)
    return escape(value) if value else None


def _escape_optional_string_into_list(arg_name):
    value = request.args.get(arg_name, default=None, type=str)
    if not value:
        return None
    return escape(value).split(',')


def _escape_optional_string_into_list_of_floats(arg_name):
    values = _escape_optional_string_into_list(arg_name)
    if not values:
        return
    return list(map(float, values))


def _escape_optional_string_into_list_of_ints(arg_name):
    values = _escape_optional_string_into_list(arg_name)
    if not values:
        return
    return list(map(int, values))


def _escape_optional_string_into_float(arg_name):
    value = _escape_optional_string(arg_name)
    return float(value) if value else None


@app.route('/')
def index():
    return 'Index Page'


@app.route('/name/<string:name1>')
def name(name1):
    data = displayer.name(
        name=escape(name1),
        after=request.args.get('after', default=None, type=int),
        before=request.args.get('before', default=None, type=int),
    )
    return jsonify(data)


@app.route('/search')
def search():
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


@app.route('/compare/<string:names>')
def compare(names):
    data = displayer.search(pattern='^({})$'.format('|'.join(escape(names).split(','))))
    return jsonify(data)


@app.route('/predict/age/<string:name1>')
def predict_age(name1):
    data = displayer.predict_age(
        name=escape(name1),
        gender=_escape_optional_string('gender'),
        exclude_deceased=request.args.get('exclude_deceased', default=False, type=bool),
        buckets=request.args.get('buckets', default=None, type=int),
    )
    return jsonify(data)


@app.route('/predict/gender/<string:name1>')
def predict_gender(name1):
    data = displayer.predict_gender(
        name=escape(name1),
        birth_year=request.args.get('birth_year', default=None, type=int),
        exclude_deceased=request.args.get('exclude_deceased', default=False, type=bool),
    )
    return jsonify(data)


if __name__ == '__main__':
    app.run()
