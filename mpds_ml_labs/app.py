
import os, sys

import ujson as json

from flask import Flask, Blueprint, Response, request, send_from_directory

from struct_utils import detect_format, poscar_to_ase, refine, get_formula
from cif_utils import cif_to_ase, ase_to_eq_cif
from prediction import get_prediction, get_aligned_descriptor, get_ordered_descriptor, get_legend, load_ml_models
from common import SERVE_UI, ML_MODELS


app_labs = Blueprint('app_labs', __name__)
static_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '../webassets'))
active_ml_models = {}

def fmt_msg(msg, http_code=400):
    return Response('{"error":"%s"}' % msg, content_type='application/json', status=http_code)

def is_plain_text(test):
    try: test.decode('ascii')
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
    def index():
        return send_from_directory(static_path, 'index.html')
    @app_labs.route('/index.css', methods=['GET'])
    def style():
        return send_from_directory(static_path, 'index.css')
    @app_labs.route('/player.html', methods=['GET'])
    def player():
        return send_from_directory(static_path, 'player.html')

@app_labs.after_request
def add_cors_header(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app_labs.route("/predict", methods=['POST'])
def predict():
    if 'structure' not in request.values:
        return fmt_msg('Invalid request')

    structure = request.values.get('structure')
    if not 0 < len(structure) < 32768:
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
