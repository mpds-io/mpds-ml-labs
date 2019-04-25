
import os, sys
import time

from struct_utils import detect_format, poscar_to_ase, refine
from cif_utils import cif_to_ase
from prediction import ase_to_prediction, load_ml_models, load_comp_models, prop_models
from common import ML_MODELS, COMP_MODELS, DATA_PATH


models, structures = [], []

if sys.argv[1:]:
    inputs = [f for f in sys.argv[1:] if os.path.isfile(f)]

    models, structures = [
        f for f in inputs if f.endswith('.pkl')
    ], [
        f for f in inputs if not f.endswith('.pkl')
    ]

if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
    target = sys.argv[1]
    structures = [os.path.join(target, f) for f in os.listdir(target) if os.path.isfile(os.path.join(target, f))]

if not models:
    models = ML_MODELS

if not structures:
    structures = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, f)) and 'settings.ini' not in f]

active_ml_models = load_ml_models(models)
if COMP_MODELS:
    active_ml_models = load_comp_models(COMP_MODELS, active_ml_models)

start_time = time.time()

for fname in structures:
    print(fname + "="*40)
    try:
        structure = open(fname).read()
    except UnicodeDecodeError as error:
        print(error)
        continue

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

    if 'disordered' not in ase_obj.info:
        ase_obj, error = refine(ase_obj)
        if error:
            print(error)
            continue

    prediction, error = ase_to_prediction(ase_obj, active_ml_models)
    if error:
        print(error)
        continue

    for prop_id, pdata in prediction.items():
        print("{0:40} = {1:6} (MAE = {2:4}), {3}".format(
            prop_models[prop_id]['name'],
            'conductor' if pdata['value'] == 0 and prop_id == 'w' else pdata['value'],
            pdata['mae'],
            prop_models[prop_id]['units']
        ))

print("Done in %1.2f sc" % time.time() - start_time)
