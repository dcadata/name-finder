from flask import Flask, request, jsonify, render_template

from names_by_peak import load_final, filter_final

app = Flask(__name__)
app.json.sort_keys = False


class AppDataset:
    names_by_peak = load_final()


@app.route('/peak', methods=['GET'])
def peak():
    return render_template('names_by_peak.html')


@app.route('/peak-api', methods=['POST'])
def peak_api():
    payload = request.json
    ageBallpark = payload.get('ageBallpark')
    neverTop = payload.get('neverTop')
    numLo = payload.get('numLo')
    numHi = payload.get('numHi')
    numResults = payload.get('numResults')

    result = filter_final(
        AppDataset.names_by_peak,
        year=int(payload.get('year')),
        yearBand=int(payload.get('yearBand')),
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
    else:
        result = [{'No results found.': ''}]
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
