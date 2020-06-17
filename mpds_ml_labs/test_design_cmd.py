
import random
import time

from struct_utils import order_disordered
from knn_sample import knn_sample
from similar_els import materialize, score_grade, score_abs
from common import connect_database, ML_MODELS
from cif_utils import ase_to_eq_cif
from prediction import prop_models, load_ml_models
from prediction_ranges import prediction_ranges, RANGE_TOLERANCE


MAX_DESIGN_MATCH = True

result, error = None, "No results (outside of prediction capabilities)"
sample = {}

for prop_id in prediction_ranges:
    dice = random.choice([0, 1])
    bound = (prediction_ranges[prop_id][1] - prediction_ranges[prop_id][0]) / 3
    if dice:
        sample[prop_id + '_min'] = float(prediction_ranges[prop_id][0] + bound * 2)
        sample[prop_id + '_max'] = float(prediction_ranges[prop_id][1])
    else:
        sample[prop_id + '_min'] = float(prediction_ranges[prop_id][0])
        sample[prop_id + '_max'] = float(prediction_ranges[prop_id][0] + bound)

range_tols = {
    prop_id: (sample[prop_id + '_max'] - sample[prop_id + '_min']) * RANGE_TOLERANCE
    for prop_id in prop_models
}

active_ml_models = load_ml_models(ML_MODELS, debug=False)

start_time = time.time()

cursor, connection = connect_database()
els_samples = knn_sample(cursor, sample)
connection.close()

results = []

if MAX_DESIGN_MATCH:
    LIMIT_TOL = 1
    while len(els_samples):

        els_sample = els_samples.pop()

        sequence, error = materialize(els_sample, active_ml_models)
        if error:
            break
        if not sequence:
            continue

        result = score_grade(sequence, sample, range_tols)
        if result['grade'] > 6:
            results.append(result)

        if len(results) > LIMIT_TOL:
            break
else:
    LIMIT_TOL = 3
    for n_attempt, els_sample in enumerate(els_samples):

        if n_attempt > LIMIT_TOL: break

        sequence, error = materialize(els_sample, active_ml_models)
        if error:
            break
        if not sequence:
            continue

        results.append(score_grade(sequence, sample, range_tols))

if not results: raise RuntimeError(error)

result = score_abs(results, sample)

answer_props = {prop_id: result['prediction'][prop_id]['value'] for prop_id in result['prediction']}

# normalization 10**5
answer_props['t'] /= 100000
sample['t_min'] /= 100000
sample['t_max'] /= 100000

if 'disordered' in result['structure'].info:
    result['structure'], error = order_disordered(result['structure'])
    if error: raise RuntimeError(error)

    result['structure'].center(about=0.0)

aux_info = []
for k, v in answer_props.items():
    aux_info.append([
        prop_models[k]['name'].replace(' ', '_'),
        sample[k + '_min'],
        v,
        sample[k + '_max'],
        prop_models[k]['units']
    ])

print(ase_to_eq_cif(result['structure'], supply_sg=False, mpds_labs_loop=[ result['grade'] ] + aux_info)[:1250])
print("Done in %1.2f sc" % (time.time() - start_time))
