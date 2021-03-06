
import os, sys

import ujson as json

from flask import Flask, Blueprint, Response, request, send_from_directory

from struct_utils import detect_format, poscar_to_ase, refine, get_formula, order_disordered
from cif_utils import cif_to_ase, ase_to_eq_cif
from prediction import prop_models, get_prediction, get_aligned_descriptor, get_ordered_descriptor, get_legend, load_ml_models, load_comp_models
from common import SERVE_UI, ML_MODELS, COMP_MODELS, connect_database
from knn_sample import knn_sample
from similar_els import materialize, score_grade, score_abs
from prediction_ranges import RANGE_TOLERANCE


__author__ = 'Evgeny Blokhin <eb@tilde.pro>'
__copyright__ = 'Copyright (c) 2018, Evgeny Blokhin, Tilde Materials Informatics'
__license__ = 'LGPL-2.1+'


app_labs = Blueprint('app_labs', __name__)
static_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '../webassets'))
active_ml_models = {}


def fmt_msg(msg, http_code=400):
    return Response('{"error":"%s"}' % msg, content_type='application/json', status=http_code)


def is_plain_text(test):
    try: test.encode('ascii')
    except: return False
    else: return True


def html_formula(string):
    sub, formula = False, ''
    for symb in string:
        if symb.isdigit() or symb == '.' or symb == '-':
            if not sub:
                formula += '<sub>'
                sub = True
        else:
            if sub and symb != 'd':
                formula += '</sub>'
                sub = False
        formula += symb
    if sub:
        formula += '</sub>'
    return formula


if SERVE_UI:
    @app_labs.route('/', methods=['GET'])
    @app_labs.route('/props.html', methods=['GET'])
    def index():
        return send_from_directory(static_path, 'props.html')

    @app_labs.route('/common.css', methods=['GET'])
    def css():
        return send_from_directory(static_path, 'common.css')

    @app_labs.route('/player.html', methods=['GET'])
    def player():
        return send_from_directory(static_path, 'player.html')

    @app_labs.route('/design.html', methods=['GET'])
    def md():
        return send_from_directory(static_path, 'design.html')

    @app_labs.route('/nouislider.min.js', methods=['GET'])
    def nouislider():
        return send_from_directory(static_path, 'nouislider.min.js')


@app_labs.after_request
def add_cors_header(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


@app_labs.route("/predict", methods=['POST'])
def predict():
    """
    A main endpoint for the properties
    prediction, based on the provided CIF
    or POSCAR
    """
    if 'structure' not in request.values:
        return fmt_msg('Invalid request')

    structure = request.values.get('structure')
    if not 0 < len(structure) < 200000:
        return fmt_msg('Request size is invalid')

    if not is_plain_text(structure):
        return fmt_msg('Request contains unsupported (non-latin) characters')

    fmt = detect_format(structure)

    if fmt == 'cif':
        ase_obj, error = cif_to_ase(structure)
        if error:
            return fmt_msg(error)

    elif fmt == 'poscar':
        ase_obj, error = poscar_to_ase(structure)
        if error:
            return fmt_msg(error)

    else: return fmt_msg('Provided data format is not supported')

    if 'disordered' in ase_obj.info:
        descriptor, error = get_ordered_descriptor(ase_obj)
        if error:
            return fmt_msg(error)

    else:
        ase_obj, error = refine(ase_obj)
        if error:
            return fmt_msg(error)

        descriptor, error = get_aligned_descriptor(ase_obj)
        if error:
            return fmt_msg(error)

    prediction, error = get_prediction(descriptor, active_ml_models)
    if error:
        return fmt_msg(error)

    if len(ase_obj) < 10:
        orig_cell = ase_obj.cell[:]
        ase_obj *= (2, 2, 2)
        ase_obj.set_cell(orig_cell)
    ase_obj.center(about=0.0)

    return Response(
        json.dumps({
            'prediction': prediction,
            'legend': get_legend(prediction),
            'formula': html_formula(get_formula(ase_obj)),
            'p1_cif': ase_to_eq_cif(ase_obj)
            }, indent=4, escape_forward_slashes=False
        ),
        content_type='application/json'
    )


@app_labs.route("/download_cif", methods=['POST'])
def download_cif():
    """
    An utility endpoint to force
    a browser file (CIF) download
    """
    structure = request.values.get('structure')
    title = request.values.get('title')

    if not structure or not title:
        return fmt_msg('Invalid request')

    if not 0 < len(structure) < 100000:
        return fmt_msg('Request size is invalid')

    return Response(structure, mimetype="chemical/x-cif", headers={
        "Content-Disposition": "attachment;filename=%s.cif" % title
    })


@app_labs.route("/design", methods=['POST'])
def design():
    """
    A main endpoint for generating
    the CIF structure based on
    the provided values of the properties
    """
    if 'numerics' not in request.values:
        return fmt_msg('Invalid request')

    try: numerics = json.loads(request.values.get('numerics'))
    except:
        return fmt_msg('Invalid request')
    if type(numerics) != dict:
        return fmt_msg('Invalid request')

    user_ranges_dict = {}

    for prop_id in prop_models:
        if prop_id not in numerics or type(numerics[prop_id]) != list or len(numerics[prop_id]) != 2:
            return fmt_msg('Invalid request')
        try: user_ranges_dict[prop_id + '_min'], user_ranges_dict[prop_id + '_max'] = float(numerics[prop_id][0]), float(numerics[prop_id][1])
        except:
            return fmt_msg('Invalid request')

    if user_ranges_dict['w_min'] == 0 and user_ranges_dict['w_max'] == 0:
        user_ranges_dict['w_min'], user_ranges_dict['w_max'] = -100, 100 # NB. any band gap is allowed

    range_tols = {
        prop_id: (user_ranges_dict[prop_id + '_max'] - user_ranges_dict[prop_id + '_min']) * RANGE_TOLERANCE
        for prop_id in prop_models
    }

    result, error = None, "No results (outside of prediction capabilities)"

    cursor, connection = connect_database()
    els_samples = knn_sample(cursor, user_ranges_dict)
    connection.close()

    results = []
    LIMIT_TOL = 1
    while len(els_samples):
        #print("TRYING TO MATERIALIZE", ", ".join(els_sample))

        els_sample = els_samples.pop()

        sequence, error = materialize(els_sample, active_ml_models)
        if error:
            break
        if not sequence:
            continue

        result = score_grade(sequence, user_ranges_dict, range_tols)
        if result['grade'] > 6:
            results.append(result)

        if len(results) > LIMIT_TOL:
            break

    if results:
        result = score_abs(results, user_ranges_dict)

        answer_props = {prop_id: result['prediction'][prop_id]['value'] for prop_id in result['prediction']}
        answer_props['t'] /= 100000 # normalization 10**5
        # NB. no scaling for *i* here

        if 'disordered' in result['structure'].info:
            result['structure'], error = order_disordered(result['structure'])
            if error:
                return fmt_msg(error)
            result['structure'].center(about=0.0)

        formula = get_formula(result['structure'])

        aux_info = []
        for k, value in answer_props.items():
            aux_info.append([
                prop_models[k]['name'].replace(' ', '_'),
                user_ranges_dict[k + '_min'],
                value,
                user_ranges_dict[k + '_max'],
                prop_models[k]['gui_units']
            ])
        return Response(
            json.dumps({
                'vis_cif': ase_to_eq_cif(
                    result['structure'],
                    supply_sg=False,
                    mpds_labs_loop=[ result['grade'] ] + aux_info
                ),
                'props': answer_props,
                'formula': html_formula(formula),
                'title': formula
                }, indent=4, escape_forward_slashes=False
            ),
            content_type='application/json'
        )

    return fmt_msg(error)


if __name__ == '__main__':
    if sys.argv[1:]:
        print("Models to load:\n" + "\n".join(sys.argv[1:]))
        active_ml_models = load_ml_models(sys.argv[1:])

    elif ML_MODELS:
        print("Models to load:\n" + "\n".join(ML_MODELS))
        active_ml_models = load_ml_models(ML_MODELS)

    else:
        print("No models to load")

    app = Flask(__name__)
    app.debug = False
    app.register_blueprint(app_labs)
    app.run()

    # NB an external WSGI-compliant server is a must
    # while exposing to the outer world

else:
    active_ml_models = load_ml_models(ML_MODELS)

if COMP_MODELS:
    active_ml_models = load_comp_models(COMP_MODELS, active_ml_models)
