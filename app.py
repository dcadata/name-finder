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

    if not year:
        result = [{'Error(s)': 'Enter year.'}]
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


@app.route('/predict-age', methods=['POST'])
def predict_age():
    payload = request.json
    name = payload.get('name')
    sex = payload.get('sex')
    mid_percentile = payload.get('mid_percentile')
    errors = []

    if not name:
        errors.append('`name` not passed')

    if not sex:
        errors.append('`sex` not passed')
    else:
        sex = sex.lower()
        if sex not in 'fm':
            errors.append('`sex` must be `f` or `m`')

    if errors:
        result = dict(errors=errors)
    else:
        kwargs = dict(name=name, sex=sex)
        if mid_percentile:
            kwargs['mid_percentile'] = float(mid_percentile)
        result = displayer.predict_age(**kwargs)
        result.percentile = result.percentile.round(3)
        result = dict(params=kwargs, data=result.to_dict('index'))

    return jsonify(result)


if __name__ == '__main__':
    displayer = Displayer()
    displayer.build_base()
    app.run()
