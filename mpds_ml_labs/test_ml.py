
import os, sys

from struct_utils import detect_format, poscar_to_ase, symmetrize
from cif_utils import cif_to_ase
from prediction import ase_to_ml_model, load_ml_model, prop_semantics
from common import ML_MODELS, DATA_PATH


models, structures = [], []

if sys.argv[1:]:
    inputs = [f for f in sys.argv[1:] if os.path.isfile(f)]
    models, structures = \
        [f for f in inputs if f.endswith('.pkl')], [f for f in inputs if not f.endswith('.pkl')]

if not models:
    models = ML_MODELS

if not structures:
    structures = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, f))]

active_ml_model = load_ml_model(models)

for fname in structures:
    print
    print(fname)
    structure = open(fname).read()

    fmt = detect_format(structure)

    if fmt == 'cif':
        ase_obj, error = cif_to_ase(structure)
        if error:
            print(error)
            continue

    elif fmt == 'poscar':
        ase_obj, error = poscar_to_ase(structure)
        if error:
            print(error)
            continue

    else:
        print('Error: %s is not a crystal structure' % fname)
        continue

    ase_obj, error = symmetrize(ase_obj)
    if error:
        print(error)
        continue

    prediction, error = ase_to_ml_model(ase_obj, active_ml_model)
    if error:
        print(error)
        continue

    for prop_id, pdata in prediction.items():
        print("{0:40} = {1:6} (MAE = {2:4}), {3}".format(
            prop_semantics[prop_id]['name'],
            pdata['value'],
            pdata['mae'],
            prop_semantics[prop_id]['units']
        ))
