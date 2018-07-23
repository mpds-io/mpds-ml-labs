
from __future__ import division
import random
import time

from struct_utils import order_disordered
from knn_sample import knn_sample
from similar_els import materialize, score
from common import connect_database, KNN_TABLE, ML_MODELS
from cif_utils import ase_to_eq_cif
from prediction import prop_models, load_ml_models
from prediction_ranges import prediction_ranges, TOL_QUALITY


result, error = None, "No results (outside of prediction capabilities)"
sample = {}

for prop_id in prediction_ranges:
    dice = random.choice([0, 1])
    bound = (prediction_ranges[prop_id][0] + prediction_ranges[prop_id][1]) / 3
    if dice:
        sample[prop_id + '_min'] = float(bound * 2)
        sample[prop_id + '_max'] = float(prediction_ranges[prop_id][1])
    else:
        sample[prop_id + '_min'] = float(prediction_ranges[prop_id][0])
        sample[prop_id + '_max'] = float(bound)

active_ml_models = load_ml_models(ML_MODELS, debug=False)

starttime = time.time()

cursor, connection = connect_database()

els_samples = knn_sample(cursor, sample)
for els_sample in els_samples:

    scoring, error = materialize(els_sample, active_ml_models)
    if error or not scoring:
        continue

    result = score(scoring, sample)
    break

connection.close()

if not result: raise RuntimeError(error)

answer_props = {prop_id: result['prediction'][prop_id]['value'] for prop_id in result['prediction']}

# normalization 10**5
answer_props['t'] /= 100000
sample['t_min'] /= 100000
sample['t_max'] /= 100000

if 'disordered' in result['structure'].info:
    result['structure'], error = order_disordered(result['structure'])
    if error: raise RuntimeError(error)

    result['structure'].center(about=0.0)

result_quality, aux_info = 0, []
for k, v in answer_props.items():
    aux_info.append([
        prop_models[k]['name'].replace(' ', '_'),
        sample[k + '_min'],
        v,
        sample[k + '_max'],
        prop_models[k]['units']
    ])
    tol = (sample[k + '_max'] - sample[k + '_min']) * TOL_QUALITY
    if sample[k + '_min'] - tol < v < sample[k + '_max'] + tol:
        result_quality += 1

print(ase_to_eq_cif(result['structure'], supply_sg=False, mpds_labs_loop=[result_quality] + aux_info))
print("Done in %1.2f sc" % (time.time() - starttime))
