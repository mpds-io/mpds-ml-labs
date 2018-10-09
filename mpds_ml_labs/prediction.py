
from __future__ import division
import os

import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix


__author__ = 'Evgeny Blokhin <eb@tilde.pro>'
__copyright__ = 'Copyright (c) 2018, Evgeny Blokhin, Tilde Materials Informatics'
__license__ = 'LGPL-2.1+'


prop_models = {
    'w': {
        'name': 'energy gap',
        'units': 'eV',
        'symbol': 'E<sub>g</sub>',
        'rounding': 1,
        'interval': [0.01, 20]
    },
    'z': {
        'name': 'isothermal bulk modulus',
        'units': 'GPa',
        'symbol': 'B<sub>T</sub>',
        'rounding': 0,
        'interval': [0.5, 2000]
    },
    'y': {
        'name': 'enthalpy of formation',
        'units': 'kJ g-at.-1',
        'symbol': '&Delta;<sub>f</sub>H',
        'rounding': 0,
        'interval': [-900, 200]
    },
    'x': {
        'name': 'heat capacity at constant pressure',
        'units': 'J K-1 g-at.-1',
        'symbol': 'C<sub>p</sub>',
        'rounding': 0,
        'interval': [0, 500]
    },
    'k': {
        'name': 'Seebeck coefficient',
        'units': 'muV K-1',
        'symbol': 'S',
        'rounding': 1,
        'interval': [-1000, 1000]
    },
    'm': {
        'name': 'temperature for congruent melting',
        'units': 'K',
        'symbol': 'T<sub>fus</sub>',
        'rounding': 0,
        'interval': [10, 5000]
    },
    'd': {
        'name': 'Debye temperature',
        'units': 'K',
        'symbol': '&Theta;<sub>D</sub>',
        'rounding': 0,
        'interval': [10, 2000]
    },
    't': {
        'name': 'linear thermal expansion coefficient',
        'units': 'K-1',
        'symbol': '&alpha;(10<sup>-5</sup>)',
        'rounding': 2,
        'interval': [-0.001, 0.001]
    }
}

periodic_elements = ['X',
                                                                                                                                                                                    'H',  'He',
'Li', 'Be',                                                                                                                                                 'B',  'C',  'N',  'O',  'F',  'Ne',
'Na', 'Mg',                                                                                                                                                 'Al', 'Si', 'P',  'S',  'Cl', 'Ar',
'K',  'Ca',                                                                                     'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
'Rb', 'Sr',                                                                                     'Y',  'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',  'Xe',
'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U',  'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

periodic_numbers = [0,
                                                                                                                                                                                    1,    112,
2,    8,                                                                                                                                                     82,   88,   94,  100,  106,  113,
3,    9,                                                                                                                                                     83,   89,   95,  101,  107,  114,
4,   10,                                                                                        14,    46,   50,   54,   58,   62,   66,   70,   74,   78,   84,   90,   96,  102,  108,  115,
5,   11,                                                                                        15,    47,   51,   55,   59,   63,   67,   71,   75,   79,   85,   91,   97,  103,  109,  116,
6,   12,    16,    18,   20,  22,   24,    26,   28,   30,   32,   34,   36,   38,   40,  42,   44,    48,   52,   56,   60,   64,   68,   72,   76,   80,   86,   92,   98,  104,  110,  117,
7,   13,    17,    19,   21,  23,   25,    27,   29,   31,   33,   35,   37,   39,   41,  43,   45,    49,   53,   57,   61,   65,   69,   73,   77,   81,   87,   93,   99,  105,  111,  118]

MIN_DESCRIPTOR_LEN = 100
N_ITER_DISORDER = 4 # the more iterations, the more consistent the ML prediction,
                    # but the more expensive the calculation


def get_descriptor(ase_obj, kappa=None, overreach=False):
    """
    From ASE object obtain
    a vectorized atomic structure
    populated to a certain fixed (relatively big) volume
    defined by kappa
    """
    if not kappa: kappa = 18
    if overreach: kappa *= 2

    norms = np.array([ np.linalg.norm(vec) for vec in ase_obj.get_cell() ])
    multiple = np.ceil(kappa / norms).astype(int)
    ase_obj = ase_obj.repeat(multiple)
    com = ase_obj.get_center_of_mass()
    ase_obj.translate(-com)
    del ase_obj[
        [atom.index for atom in ase_obj if np.sqrt(np.dot(atom.position, atom.position)) > kappa]
    ]

    ase_obj.center()
    ase_obj.set_pbc((False, False, False))
    sorted_seq = np.argsort(np.fromiter((np.sqrt(np.dot(x, x)) for x in ase_obj.positions), np.float))
    ase_obj = ase_obj[sorted_seq]

    elements, positions = [], []
    for atom in ase_obj:
        elements.append(periodic_numbers[periodic_elements.index(atom.symbol)] - 1)
        positions.append(int(round(np.sqrt(atom.position[0]**2 + atom.position[1]**2 + atom.position[2]**2) * 10)))

    return np.array([elements, positions])


def get_ordered_descriptor(ase_obj, kappa=None, overreach=False):
    if 'disordered' not in ase_obj.info:
        return None, "Expected disordered structure, got ordered structure"

    from struct_utils import order_disordered

    descriptor = None
    for _ in range(N_ITER_DISORDER):
        order_obj, error = order_disordered(ase_obj)
        if error: return None, error

        interim_descriptor = get_descriptor(order_obj, kappa=kappa, overreach=overreach)
        if len(interim_descriptor[0]) < MIN_DESCRIPTOR_LEN:
            if overreach:
                return None, "Cannot get proper descriptor"

            interim_descriptor = get_descriptor(order_obj, kappa=kappa, overreach=True)
            if len(interim_descriptor[0]) < MIN_DESCRIPTOR_LEN:
                return None, "Cannot get proper descriptor"

        if descriptor is not None:
            left_len, right_len = len(descriptor[0]), len(interim_descriptor[0])

            if left_len != right_len: # align length
                descriptor =         descriptor[:, :min(left_len, right_len)]
                interim_descriptor = interim_descriptor[:, :min(left_len, right_len)]

            descriptor = (descriptor + interim_descriptor)/2
        else:
            descriptor = interim_descriptor[:]

    return descriptor, None


def get_aligned_descriptor(ase_obj, kappa=None):

    descriptor = get_descriptor(ase_obj, kappa=kappa)
    if len(descriptor[0]) < MIN_DESCRIPTOR_LEN:
        descriptor = get_descriptor(ase_obj, kappa=kappa, overreach=True)
        if len(descriptor[0]) < MIN_DESCRIPTOR_LEN:
            return None, "Cannot get proper descriptor"

    return descriptor, None


def load_ml_models(prop_model_files, debug=True):

    import cPickle

    ml_models = {}
    for file_name in prop_model_files:
        if not os.path.exists(file_name):
            if debug: print("No file %s" % file_name)
            continue

        basename = file_name.split(os.sep)[-1]
        if basename.startswith('ml') and basename[3:4] == '_':
            prop_id = basename[2:3]
            if debug:
                if prop_id in prop_models:
                    print("Detected regressor model-%s <%s> in file %s" % (prop_id, prop_models[prop_id]['name'], basename))
                else:
                    print("Detected model-%s in file %s" % (prop_id, basename))

        else: raise RuntimeError("Unknown model file: %s" % basename)

        with open(file_name, 'rb') as f:
            model = cPickle.load(f)
            if hasattr(model, 'predict') and hasattr(model, 'metadata'):
                ml_models[prop_id] = model
                if debug: print("Model-%s %s metadata: %s" % (prop_id, basename, model.metadata))

    if debug: print("Loaded property models: %s" % len(ml_models))
    return ml_models


def get_legend(pred_dict):
    legend = {}
    for key in pred_dict.keys():
        legend[key] = prop_models.get(key, {
            'name': 'Unspecified property ' + str(key),
            'units': 'arb.u.',
            'symbol': 'P' + str(key),
            'rounding': 0
        })
    return legend


def ase_to_prediction(ase_obj, ml_models, prop_ids=False):
    """
    Higher-level prediction handler that is able to
    resolve disordered structures

    Returns:
        Prediction (dict) *or* None
        None *or* error (str)
    """
    if 'disordered' in ase_obj.info:

        from struct_utils import order_disordered

        results, avg_results = {}, {}
        for _ in range(N_ITER_DISORDER):
            order_obj, error = order_disordered(ase_obj)
            if error:
                return None, error

            sample, error = ase_to_prediction(order_obj, ml_models, prop_ids)
            if error:
                return None, error

            for prop_id, pdata in sample.items():
                avg_results.setdefault(prop_id, []).append(pdata['value'])

        for prop_id, values in avg_results.items():
            if prop_id == 'w' and values.count(0) == 1: # considering classifier error
                values.remove(0)

            results[prop_id] = {
                'value': round(np.median(values), prop_models[prop_id]['rounding']),
                'mae': round(ml_models[prop_id].metadata['mae'], prop_models[prop_id]['rounding']),
                'r2': ml_models[prop_id].metadata['r2']
            }

        return results, None

    descriptor, error = get_aligned_descriptor(ase_obj)
    if error:
        return None, error

    return get_prediction(descriptor, ml_models, prop_ids)


def get_prediction(descriptor, ml_models, prop_ids=False):
    """
    Execute all the regressor models against a given structure descriptor;
    the results of the "w" regressor model will depend on
    the output of the "0" binary classifier model

    Returns:
        Prediction (dict) *or* None
        None *or* error (str)
    """
    if not prop_ids:
        prop_ids = ml_models.keys()

    if type(prop_ids) != list:
        prop_ids = prop_ids.split()

    # testing
    if not ml_models:
        return {prop_id: {'value': 0, 'mae': 0, 'r2': 0} for prop_id in prop_ids}, None

    if not set(prop_ids).issubset(ml_models.keys()):
        return None, 'Unrecognized model: ' + ', '.join(prop_ids)

    result = {}
    descriptor = descriptor.flatten()
    d_dim = len(descriptor)

    if 'w' in prop_ids: # classifier invocation
        if '0' not in ml_models:
            return None, 'Classifier model is required but not available'
        if '0' not in prop_ids:
            prop_ids.append('0')

    # production
    for prop_id in prop_ids:

        if prop_id == 'w' and result.get('w', {}).get('value') == 0:
            continue

        if d_dim < ml_models[prop_id].n_features_:
            continue
        elif d_dim > ml_models[prop_id].n_features_:
            d_input = descriptor[:ml_models[prop_id].n_features_]
        else:
            d_input = descriptor[:]

        try:
            prediction = ml_models[prop_id].predict([d_input])[0]
        except Exception as e:
            return None, str(e)

        if prop_id == '0':
            if prediction == 0:
                result['w'] = {'value': 0, 'mae': 0, 'r2': 0}

        else:
            result[prop_id] = {
                'value': round(prediction, prop_models[prop_id]['rounding']),
                'mae': round(ml_models[prop_id].metadata['mae'], prop_models[prop_id]['rounding']),
                'r2': ml_models[prop_id].metadata['r2']
            }

    return result, None


def get_regr(a=None, b=None):

    if not a: a = 100
    if not b: b = 2

    return RandomForestRegressor(
        n_estimators=a,
        max_features=b,
        max_depth=None,
        min_samples_split=2, # recommended value
        min_samples_leaf=5, # recommended value
        bootstrap=True, # recommended value
        n_jobs=-1
    )


def get_clfr(a=None, b=None):

    if not a: a = 100
    if not b: b = 2

    return RandomForestClassifier(
        n_estimators=a,
        max_features=b,
        max_depth=None,
        min_samples_split=2, # recommended value
        min_samples_leaf=5, # recommended value
        bootstrap=True, # recommended value
        n_jobs=-1
    )


def estimate_regr_quality(algo, args, values, attempts=30, nsamples=0.33):

    results = []

    for _ in range(attempts):
        X_train, X_test, y_train, y_test = train_test_split(args, values, test_size=nsamples)
        algo.fit(X_train, y_train)

        prediction = algo.predict(X_test)

        mae = mean_absolute_error(y_test, prediction)
        r2 = r2_score(y_test, prediction)
        results.append([mae, r2])

    results = list(map(list, zip(*results))) # transpose

    avg_mae = np.median(results[0])
    avg_r2 = np.median(results[1])
    return avg_mae, avg_r2


def estimate_clfr_quality(algo, args, values, attempts=30, nsamples=0.33):

    results = []

    for _ in range(attempts):
        X_train, X_test, y_train, y_test = train_test_split(args, values, test_size=nsamples)
        algo.fit(X_train, y_train)

        prediction = algo.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
        error_percentage = (fp + fn)/(tn + fp + fn + tp)
        results.append(error_percentage)

    return np.median(results)
