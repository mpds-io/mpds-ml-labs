
import os, sys

import ujson as json

from flask import Flask, Blueprint, Response, request, send_from_directory

from cors import crossdomain
from struct_utils import detect_format, poscar_to_ase, symmetrize
from cif_utils import cif_to_ase, ase_to_eq_cif
from prediction import ase_to_ml_model, get_legend, load_ml_model


app_labs = Blueprint('app_labs', __name__)
ml_model = None

def fmt_msg(msg, http_code=400):
    return Response('{"error":"%s"}' % msg, content_type='application/json', status=http_code)

def is_plain_text(test):
    try: test.decode('ascii')
    except: return False
    else: return True

@app_labs.route('/', methods=['GET'])
def index():
    return send_from_directory(os.path.dirname(__file__), 'index.html')

@app_labs.route('/index.css', methods=['GET'])
def style():
    return send_from_directory(os.path.dirname(__file__), 'index.css')

@app_labs.route('/player.html', methods=['GET'])
def player():
    return send_from_directory(os.path.dirname(__file__), 'player.html')

@app_labs.route("/predict", methods=['POST'])
@crossdomain(origin='*')
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

    else:
        return fmt_msg('Provided data format is not supported')

    ase_obj, error = symmetrize(ase_obj)
    if error:
        return fmt_msg(error)

    prediction, error = ase_to_ml_model(ase_obj, ml_model)
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
            'p1_cif': ase_to_eq_cif(ase_obj)
            }, indent=4, escape_forward_slashes=False
        ),
        content_type='application/json'
    )


if __name__ == '__main__':
    if sys.argv[1:]:
        ml_model = load_ml_model(sys.argv[1:])
        print("Loaded models: " + " ".join(sys.argv[1:]))
    else:
        print("No model loaded")

    app = Flask(__name__)
    app.debug = False
    app.register_blueprint(app_labs)
    app.run()

    # NB an external WSGI-compliant server is a must
    # while exposing to the outer world
