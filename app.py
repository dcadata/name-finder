from flask import Flask, request, jsonify
from markupsafe import escape

import core

displayer = core.Displayer()
displayer.load()

app = Flask(__name__)


def _escape_optional_string(arg_name: str):
    value = request.args.get(arg_name, default=None, type=str)
    if not value:
        return
    return escape(value)


def _escape_optional_string_into_list(arg_name: str):
    value = request.args.get(arg_name, default=None, type=str)
    if not value:
        return
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


def _get_name(name: str):
    data = displayer.name(
        name=escape(name),
        after=request.args.get('after', default=None, type=int),
        before=request.args.get('before', default=None, type=int),
        year=request.args.get('year', default=None, type=int),
        show_historic=True,
        n_bars=30,
    )
    n = data.get('name')
    html = open('templates/name_page.html').read().format(
        name=data['name'] if n else name.title(),
        info=''.join((f'<li>{line}</li>' for line in data['display']['info'])) if n else 'This name was not found.',
        number_bars=''.join((f'{line}<br>' for line in data['display']['number_bars'])) if n else '',
        ratio_bars=''.join((f'{line}<br>' for line in data['display']['ratio_bars'])) if n else '',
        number_bars_header_text=displayer.number_bars_header_text if n else '',
        ratio_bars_header_text=displayer.ratio_bars_header_text if n and data['display']['ratio_bars'] else '',
    )
    return html


def _get_search_by_text(query: str):
    escaped_query = escape(query)
    top = request.args.get('top', type=int)
    data = displayer.search_by_text(escaped_query, top=top)
    html = open('templates/query_page.html').read().format(query=escaped_query, top=top, info=''.join(
        '<li><a href="/?n={name}">{name}</a> {display}</li>'.format(**i) for i in data))
    return html


@app.route('/')
def home():
    if name_query := request.args.get('n', type=str):
        return _get_name(name_query)
    elif search_by_text_query := request.args.get('q', type=str):
        return _get_search_by_text(search_by_text_query)
    else:
        return open('templates/query_page.html').read()


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
    )
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
        after=request.args.get('after', default=None, type=int),
        before=request.args.get('before', default=None, type=int),
        year=request.args.get('year', default=None, type=int),
        living=bool(request.args.get('living', default=0, type=int)),
    )
    return jsonify(data)


if __name__ == '__main__':
    app.run()
