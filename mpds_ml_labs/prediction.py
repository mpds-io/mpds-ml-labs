
from __future__ import division
import os
import cPickle

import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix


prop_models = {
    'w': {
        'name': 'band gap',
        'units': 'eV',
        'symbol': 'e<sub>dir. or indir.</sub>',
        'rounding': 1,
        'interval': [0.01, 20]
    },
    'z': {
        'name': 'isothermal bulk modulus',
        'units': 'GPa',
        'symbol': 'B',
        'rounding': 0,
        'interval': [0.5, 2000]
    },
    'y': {
        'name': 'enthalpy of formation',
        'units': 'kJ g-at.-1',
        'symbol': '&Delta;H',
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
        'symbol': 'T<sub>melt</sub>',
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
        'symbol': '&Theta;<sub>D</sub>(10<sup>5</sup>)',
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

pmin, pmax = 1, max(periodic_numbers)
periodic_numbers_normed = [(i - pmin)/(pmax - pmin) for i in periodic_numbers]


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
    com = ase_obj.get_center_of_mass() # NB use recent ase version here, because of the new element symbols
    ase_obj.translate(-com)
    del ase_obj[
        [atom.index for atom in ase_obj if np.sqrt(np.dot(atom.position, atom.position)) > kappa]
    ]

    ase_obj.center()
    ase_obj.set_pbc((False, False, False))
    sorted_seq = np.argsort(np.fromiter((np.sqrt(np.dot(x, x)) for x in ase_obj.positions), np.float))
    ase_obj = ase_obj[sorted_seq]

    DV = []
    for atom in zip(
        ase_obj.get_chemical_symbols(),
        ase_obj.get_scaled_positions()
    ):
        DV.append([
            periodic_numbers_normed[periodic_elements.index(atom[0])],
            np.sqrt(atom[1][0]**2 + atom[1][1]**2 + atom[1][2]**2)
        ])

    return np.array(DV).flatten()


def load_ml_models(prop_model_files):
    ml_models = {}
    for n, file_name in enumerate(prop_model_files, start=1):
        if not os.path.exists(file_name):
            print("No file %s" % file_name)
            continue

        basename = file_name.split(os.sep)[-1]
        if basename.startswith('ml') and basename[3:4] == '_' and basename[2:3] in prop_models:
            prop_id = basename[2:3]
            print("Detected property %s in file %s" % (prop_models[prop_id]['name'], basename))
        else:
            prop_id = str(n)
            print("No property name detected in file %s" % basename)

        with open(file_name, 'rb') as f:
            model = cPickle.load(f)
            if hasattr(model, 'predict') and hasattr(model, 'metadata'):
                ml_models[prop_id] = model
                print("Model metadata: %s" % model.metadata)

    print("Loaded property models: %s" % len(ml_models))
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


def ase_to_prediction(ase_obj, ml_models):
    """
    Execute all the regressor models againts a given structure desriptor;
    the results of the "w" regressor model will depend on the
    output of the binary classifier model
    """
    result = {}
    descriptor = get_descriptor(ase_obj, overreach=True)
    d_dim = len(descriptor)
    should_invoke_clfr = 'w' in prop_models.keys()

    # testing
    if not ml_models:
        result = {prop_id: {'value': 42, 'mae': 0, 'r2': 0} for prop_id in prop_models.keys()}

        if should_invoke_clfr:
            result['w'] = {'value': 0, 'mae': 0, 'r2': 0}

    # production
    for prop_id, model in ml_models.items():

        if d_dim < model.n_features_:
            continue
        elif d_dim > model.n_features_:
            d_input = descriptor[:model.n_features_]
        else:
            d_input = descriptor[:]

        try:
            prediction = model.predict([d_input])[0]
        except Exception as e:
            return None, str(e)

        # classifier
        if model.metadata.get('error_percentage'):

            if should_invoke_clfr:

                if prediction == 0:
                    result['w'] = {'value': 0, 'mae': 0, 'r2': 0}

        # regressor
        else:
            if prop_id not in prop_models or \
            (prop_id == 'w' and prop_id in result):
                continue

            result[prop_id] = {
                'value': round(prediction, prop_models[prop_id]['rounding']),
                'mae': round(model.metadata['mae'], prop_models[prop_id]['rounding']),
                'r2': model.metadata['r2']
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
