import datetime

from flask import Flask, request, jsonify, render_template

from core import Displayer
from names_by_peak import load_final, filter_final
from predict_gender import predict_gender_batch

app = Flask(__name__)
app.json.sort_keys = False


class AppDataset:
    names_by_peak = load_final()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/peak', methods=['GET', 'POST'])
def peak():
    if request.method == 'GET':
        return render_template('names_by_peak.html')

    payload = request.json
    year = payload.get('year')
    yearBand = payload.get('yearBand')
    ageBallpark = payload.get('ageBallpark')
    neverTop = payload.get('neverTop')
    numLo = payload.get('numLo')
    numHi = payload.get('numHi')
    numResults = payload.get('numResults')

    result = [{'No results found.': ''}]

    if not year:
        return result
    if year and not yearBand:
        yearBand = 0

    result = filter_final(
        AppDataset.names_by_peak,
        year=int(year),
        yearBand=int(yearBand),
        usePeak=payload.get('usePeak'),
        ageBallpark=int(ageBallpark) if ageBallpark else None,
        sex=payload.get('sex'),
        genderCat=tuple(filter(None, [(i if payload.get(f'genderCat{i}') else None) for i in (
            'Masc', 'NeutMasc', 'Neut', 'NeutFem', 'Fem')])),
        neverTop=int(neverTop) if neverTop else None,
        numLo=int(numLo) if numLo else None,
        numHi=int(numHi) if numHi else None,
    )
    if len(result):
        result = result.iloc[:int(numResults)].to_dict('records')
    return jsonify(result)


@app.route('/predict-gender', methods=['POST'])
def predict_gender():
    result = predict_gender_batch(request.json, displayer=displayer)
    return jsonify(result)


@app.route('/predict-age', methods=['GET'])
def predict_age():
    result = dict(params=dict())
    params = request.args

    if name := params.get('name'):
        result['params'].update(dict(name=name))
    else:
        result['error'] = '`name` not passed'

    if sex := params.get('sex'):
        sex = sex.lower()
        if sex in 'fm':
            result['params'].update(dict(sex=sex))
        else:
            result['error'] = '`sex` must be `f` or `m`'
    else:
        result['error'] = '`sex` not passed'

    if mid_percentile := params.get('mid_percentile'):
        result['params'].update(dict(mid_percentile=float(mid_percentile)))

    result_as_df = displayer.predict_age(**result['params'])
    result_as_df.percentile = result_as_df.percentile.round(3)
    result_as_df['age'] = datetime.date.today().year - result_as_df.year
    result.update(result_as_df.iloc[:2].to_dict('index'))
    return jsonify(result)


if __name__ == '__main__':
    displayer = Displayer()
    displayer.build_base()
    app.run(debug=True)
