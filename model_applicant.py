
import os
import sys

import numpy as np
import treelite.runtime

from mpds_ml_labs.prediction import load_ml_models, prop_models, ase_to_prediction, get_aligned_descriptor
from mpds_ml_labs.common import ML_MODELS, DATA_PATH
from mpds_ml_labs.struct_utils import detect_format, poscar_to_ase, refine
from mpds_ml_labs.cif_utils import cif_to_ase


mod_path = '/data/ml_models_new'
mod_basename = '_cmpld.so'

COMPILED_MODELS = [mod_path + os.sep + prop_id + mod_basename for prop_id in list(prop_models.keys()) + ['0']]

print("Sklearn models to load:\n" + "\n".join(ML_MODELS))
active_sk_models = load_ml_models(ML_MODELS)

print("Treelite models to load:\n" + "\n".join(COMPILED_MODELS))
active_comp_models = {}
for modfile in COMPILED_MODELS:
    prop_id = modfile.split('/')[-1][:1]
    if prop_id not in active_sk_models:
        print('Unknown model file: %s' % modfile)
        continue
    active_comp_models[prop_id] = treelite.runtime.Predictor(modfile, verbose=False)
    active_comp_models[prop_id].metadata = active_sk_models[prop_id].metadata
    active_comp_models[prop_id].n_features_ = active_sk_models[prop_id].n_features_
    active_comp_models[prop_id].treelite = True

assert len(active_sk_models) == len(active_comp_models)

structures = []

if sys.argv[1:]:
    inputs = [f for f in sys.argv[1:] if os.path.isfile(f)]
    structures = [
        f for f in inputs if f[-3:] not in ['pkl', '.so']
    ]

if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
    structures = [os.path.join(sys.argv[1], f) for f in os.listdir(sys.argv[1]) if os.path.isfile(os.path.join(sys.argv[1], f))]

if not structures:
    structures = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, f)) and 'settings.ini' not in f]

########################################################################
for fname in structures:
    print(fname + "="*40)
    structure = open(fname).read()
    fmt = detect_format(structure)

    if fmt == 'cif':
        ase_obj, error = cif_to_ase(structure)
        assert not error, error

    elif fmt == 'poscar':
        ase_obj, error = poscar_to_ase(structure)
        assert not error, error

    else:
        print('Error: %s is not a crystal structure' % fname)
        continue

    if 'disordered' not in ase_obj.info:
        ase_obj, error = refine(ase_obj)
        assert not error, error

    prediction_sk, error = ase_to_prediction(ase_obj, active_sk_models)
    assert not error, error
    prediction_comp, error = ase_to_prediction(ase_obj, active_comp_models)
    assert not error, error

    for prop_id in prediction_sk:
        diff = abs(prediction_sk[prop_id]['value'] - prediction_comp[prop_id]['value'])
        if diff == 0:
            print("Model %s is perfect" % prop_id)
        elif diff < abs(prediction_sk[prop_id]['value']) / 50: # 2%
            print("Model %s is okayish" % prop_id)
        else:
            print('Model %s is inconsistent: %s vs. %s' % (
                prop_id, prediction_sk[prop_id]['value'], prediction_comp[prop_id]['value']
            ))
