from flask import Flask, request, jsonify
from markupsafe import escape

import finder

finder.OUTPUT_RECORDS = True

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


def _get_name(name: str) -> dict:
    data = displayer.name(
        name=escape(name),
        after=request.args.get('after', default=None, type=int),
        before=request.args.get('before', default=None, type=int),
        year=request.args.get('year', default=None, type=int),
        show_historic=bool(request.args.get('show_historic', default=0, type=int)),
    )
    return data


@app.route('/name/<string:name>')
def name_endpoint(name: str):
    return jsonify(_get_name(name))


@app.route('/compare/<string:names>')
def compare_endpoint(names: str):
    names = escape(names).split('-')
    data = [_get_name(name) for name in names]
    data = [i for i in data if i]
    data = dict(data=data, display='\n\n'.join(i['display'] for i in data))
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
        year=request.args.get('year', default=None, type=int),
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
    data = displayer.search_by_text(text=escape(text))
    return jsonify(data)


@app.route('/predict/age/<string:name>')
def predict_age_endpoint(name: str):
    data = displayer.predict_age(
        name=escape(name),
        gender=_escape_optional_string('gender'),
        living=bool(request.args.get('living', default=0, type=int)),
        buckets=request.args.get('buckets', default=None, type=int),
    )
    return jsonify(data)


@app.route('/predict/gender/<string:name>')
def predict_gender_endpoint(name: str):
    data = displayer.predict_gender(
        name=escape(name),
        year=request.args.get('year', default=None, type=int),
        living=bool(request.args.get('living', default=0, type=int)),
    )
    return jsonify(data)


if __name__ == '__main__':
    app.run()
