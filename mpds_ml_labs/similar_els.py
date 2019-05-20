
from __future__ import division
import logging
from pprint import pprint
try: from urllib.parse import urlencode
except ImportError: from urllib import urlencode

import json
import httplib2

from prediction import prop_models, periodic_elements, periodic_numbers, ase_to_prediction
from prediction_ranges import prediction_ranges, RANGE_TOLERANCE
from struct_utils import json_to_ase
from common import API_KEY, ELS_ENDPOINT


__author__ = 'Evgeny Blokhin <eb@tilde.pro>'
__copyright__ = 'Copyright (c) 2018, Evgeny Blokhin, Tilde Materials Informatics'
__license__ = 'LGPL-2.1+'


network = httplib2.Http()

normalized_f = {
    prop_id: lambda x: (x - bound[0]) / (bound[1] - bound[0])
    for prop_id, bound in prediction_ranges.items()
}


def get_similar_els(els):
    """
    Obtain chemically similar elements
    according to similarity of the periodic numbers (PNs)

    Returns:
        elements (list or None)
        error (None or string)
    """
    try: pns = [periodic_numbers[periodic_elements.index(el.capitalize())] for el in els]
    except ValueError:
        return None, "Unsupported chemical elements!"

    result, metrics = [], len(els)

    for m in [1, 0]:
        for n in range(len(pns)):
            if pns[n] in [1, 8, 14, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 88, 94, 100, 106, 112]: # upper deck
                dcf = 1

            elif pns[n] in [18, 19]: # f-elements
                dcf = 2

            elif pns[n] in [44, 45]: # f-elements
                dcf = -2

            elif pns[n] in [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43]: # f-elements
                dcf = 1 if m == 0 else -2

            elif pns[n] in [2, 9, 15, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 89, 95, 101, 107, 113]: # pre-upper deck
                dcf = 0.5 if m == 0 else -1

            else:
                dcf = -1

            new_pns = pns[:]
            new_pns[n] += dcf * (2 - m)

            try: new_els = [periodic_elements[periodic_numbers.index(el)] for el in new_pns]
            except ValueError: continue

            if len(set(new_els)) != metrics: continue

            result.append(new_els)

    if not result:
        return None, "No matches by chemical elements!"

    return result, None


def get_similar_structs(els_comb):
    """
    Employ an *els_comb* MPDS API method,
    returning the MPDS S-entries,
    composed by the given chemical elements
    """
    if not ELS_ENDPOINT:
        return None, 'No similarity search endpoint defined'

    response, content = network.request(
        uri=ELS_ENDPOINT + '?' + urlencode({
            'input': json.dumps(els_comb)
        }),
        method='GET',
        headers={'Key': API_KEY}
    )

    if response.status != 200:
        return None, 'While similarity search an HTTP error %s occured' % response.status

    try:
        content = json.loads(content)
    except:
        return None, 'Unreadable data obtained'

    return content, None


def get_group(z):
    if z == 1:
        return 1
    if z == 2:
        return 18
    if 3 <= z <= 18:
        if (z - 2) % 8 == 0:
            return 18
        elif (z - 2) % 8 <= 2:
            return (z - 2) % 8
        else:
            return 10 + (z - 2) % 8
    if 19 <= z <= 54:
        if (z - 18) % 18 == 0:
            return 18
        else:
            return (z - 18) % 18
    if (z - 54) % 32 == 0:
        return 18
    elif (z - 54) % 32 >= 17:
        return (z - 54) % 32 - 14
    else:
        return (z - 54) % 32


def compact_by_disorder(given_els):
    """
    Given a list of the chemical elements,
    reduce it, popping out chemically similar elements
    as candidates to form the structural disorder
    (cf. partial occupancies)

    Returns:
        New compacted els (list)
        New occs (dict): e.g. {el_from_compacted_els: [another_el, ..., without_el_from_compacted_els], ...}
    """
    groups = [get_group(periodic_elements.index(el)) for el in given_els]

    if len(set(groups)) == len(given_els):
        return given_els, {}

    families, compacted_els, new_occs = {}, [], {}
    for n, group in enumerate(groups):
        families.setdefault(group, []).append(given_els[n])

    for disordered_els in families.values():

        disordered_els.sort(key=lambda el: periodic_numbers[periodic_elements.index(el)])

        compacted_els.append(disordered_els.pop(0))
        if disordered_els:
            new_occs[compacted_els[-1]] = disordered_els

    return compacted_els, new_occs


def materialize(given_els, active_ml_models):
    """
    Given a list of the chemical elements,
    get scored crystal structures, having either
    exactly these elements or chemically similar elements
    """
    compacted_els, new_occs = compact_by_disorder(given_els)

    els_comb, error = get_similar_els(compacted_els)
    if error:
        return None, error

    els_comb.append(compacted_els)

    sequence, error = massage_by_similarity(els_comb, compacted_els, new_occs, active_ml_models)
    if error:
        return None, error

    if not sequence:
        for child in els_comb:

            grand_child_els_comb, error = get_similar_els(child)
            if error:
                return None, error

            sequence, error = massage_by_similarity(grand_child_els_comb, compacted_els, new_occs, active_ml_models)
            if error:
                return None, error

            if sequence:
                break

    if not sequence:
        return None, "No results (cannot compile crystal structure)"

    return sequence, None


def massage_by_similarity(input_els_comb, ref_els, ref_occs, active_ml_models):

    rows, error = get_similar_structs(input_els_comb)
    if error:
        return None, error

    sequence = []

    for row in rows:
        els_were = list(set(ref_els) - set(row['els_noneq']))
        els_are = list(set(row['els_noneq']) - set(ref_els))

        if len(els_were) != len(els_are):
            logging.error("Error: there were els: %s; there are els: %s (entry %s)" % ('-'.join(els_were), '-'.join(els_are), row['entry']))
            continue

        els_were.sort(key=lambda x: periodic_numbers[periodic_elements.index(x)])
        els_are.sort(key=lambda x: periodic_numbers[periodic_elements.index(x)])
        replacements = dict(zip(els_are, els_were))

        new_els = []
        for n, el in enumerate(row['els_noneq']):
            if el in replacements:
                new_el = replacements[el]
            else:
                new_el = el

            new_els.append(new_el)

            for another_el in ref_occs.get(new_el, []):
                new_els.append(another_el)
                row['basis_noneq'].append(row['basis_noneq'][n])
                row['occs_noneq'][n] /= (len(ref_occs[new_el]) + 1)
                row['occs_noneq'].append(row['occs_noneq'][n])

        ase_obj, error = json_to_ase([row['occs_noneq'], row['cell_abc'], row['sg_n'], row['basis_noneq'], new_els])
        if error:
            return None, error

        prediction, error = ase_to_prediction(ase_obj, active_ml_models)
        if error:
            return None, error

        sequence.append({"structure": ase_obj, "prediction": prediction})

    return sequence, None


def score_abs(sequence, prop_ranges_dict):

    if len(sequence) == 1:
        return sequence[0]

    ideal_predictions = {prop_id: (prop_ranges_dict[prop_id + '_min'] + prop_ranges_dict[prop_id + '_max']) / 2 for prop_id in prop_models}

    sequence.sort(key=lambda x: sum([
        normalized_f[prop_id]( abs(ideal_predictions[prop_id] - x["prediction"][prop_id]["value"]) ) for prop_id in prop_models
    ]))

    #print("*"*50)
    #pprint(ideal_predictions)
    #print("vs.")
    #pprint(sequence)
    #print("*"*50)

    return sequence[0]


def score_grade(sequence, prop_ranges_dict, range_tols):

    assert range_tols

    for n in range(len(sequence)):

        assert sequence[n]['prediction']
        sequence[n]["grade"] = 0

        for prop_id in prop_models:

            if prop_ranges_dict[prop_id + '_min'] - range_tols[prop_id] < sequence[n]['prediction'][prop_id]['value'] < prop_ranges_dict[prop_id + '_max'] + range_tols[prop_id]:
                sequence[n]["grade"] += 1

    sequence.sort(key=lambda x: x["grade"], reverse=True)
    return sequence[0]


if __name__ == "__main__":

    print(score_grade([
        {'prediction':{
            'z': {'value': 400},
            'y': {'value': 16},
            'x': {'value': 11},
            'k': {'value': 0},
            'w': {'value': 2},
            'm': {'value': 1000},
            'd': {'value': 800},
            't': {'value': 20},
            'i': {'value': 1000},
            'o': {'value': 1}
        }},
        {'prediction':{
            'z': {'value': 401},
            'y': {'value': 2},
            'x': {'value': 10},
            'k': {'value': 0.2},
            'w': {'value': 25},
            'm': {'value': 1000},
            'd': {'value': 800},
            't': {'value': 2},
            'i': {'value': 1000},
            'o': {'value': 1}
        }},
        {'prediction':{
            'z': {'value': 400},
            'y': {'value': -30},
            'x': {'value': 11},
            'k': {'value': -10},
            'w': {'value': 50},
            'm': {'value': 2200},
            'd': {'value': 915},
            't': {'value': 21},
            'i': {'value': 1000},
            'o': {'value': 1}
        }}
    ], {
        'z_min': 399, 'z_max': 401,
        'y_min': 0, 'y_max': 15,
        'x_min': -20, 'x_max': 20,
        'k_min': 0, 'k_max': 0.1,
        'w_min': -100, 'w_max': 100,
        'm_min': 2000, 'm_max': 3000,
        'd_min': 900, 'd_max': 950,
        't_min': -10, 't_max': -9,
        'i_min': 1200, 'i_max': 1400,
        'o_min': -10, 'o_max': 0
    }))

    print(compact_by_disorder(['Si', 'Cr', 'Mo', 'W', 'O']))
    print(compact_by_disorder(['Li', 'O', 'Mn', 'B', 'Fr', 'Re', 'Tc', 'Ga', 'Ra', 'Al']))
    print(compact_by_disorder(['V', 'Cr', 'Mn', 'Rn']))
    print(compact_by_disorder(['Cl', 'Na', 'Dy']))

    import random

    sample = {}
    for prop_id in prediction_ranges:
        dice = random.choice([0, 1])
        bound = (prediction_ranges[prop_id][1] - prediction_ranges[prop_id][0]) / 4
        if dice:
            sample[prop_id + '_min'] = prediction_ranges[prop_id][0] + bound * 3
            sample[prop_id + '_max'] = prediction_ranges[prop_id][1]
        else:
            sample[prop_id + '_min'] = prediction_ranges[prop_id][0]
            sample[prop_id + '_max'] = prediction_ranges[prop_id][0] + bound

    range_tols = {
        prop_id: (sample[prop_id + '_max'] - sample[prop_id + '_min']) * RANGE_TOLERANCE
        for prop_id in prop_models
    }

    def gen_mockup_result():
        return {'prediction': {
            prop_id: {'value': random.uniform(*bounds)} for prop_id, bounds in prediction_ranges.items()
        }}

    results = [gen_mockup_result() for _ in range(500)]
    #print(score_abs(results, sample))
    print(score_grade(results, sample, range_tols))
