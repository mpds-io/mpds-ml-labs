
from __future__ import division
import os, sys
import time
import json
from pprint import pprint

import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from mpds_client import MPDSExport

from mpds_ml_labs.prediction import prop_models, estimate_regr_quality


def get_regr(params={}):
    return RandomForestRegressor(**params)

param_dist = {
    "n_estimators": range(20, 501, 20),
    "max_features": range(5, 31) + [35, 40],
    "max_depth": [None, 10, 25, 50, 75],
    "min_samples_split": [2, 4, 10],
    "min_samples_leaf": [1, 3, 5, 7, 14],
    "bootstrap": [True, False],
    "criterion": ['mae'],
    "n_jobs": [2]
}

if __name__ == "__main__":
    try:
        data_file = sys.argv[1]
    except IndexError:
        raise RuntimeError

    if not os.path.exists(data_file):
        raise RuntimeError

    basename = data_file.split(os.sep)[-1]
    if basename.startswith('df') and basename[3:4] == '_' and basename[2:3] in prop_models:
        tag = basename[2:3]
        print("Detected property %s" % prop_models[tag]['name'])
    else:
        raise RuntimeError("No property name detected")

    df = pd.read_pickle(data_file)

    X = np.array(df['Descriptor'].tolist())
    y = df['Avgvalue'].tolist()

    starttime = time.time()

    search = RandomizedSearchCV(get_regr(), param_distributions=param_dist, n_iter=5000, cv=2, verbose=2)
    search.fit(X, y)
    avg_mae, avg_r2 = estimate_regr_quality(get_regr(search.best_params_), X, y)

    print("Avg. MAE: %.2f" % avg_mae)
    print("Avg. R2 score: %.2f" % avg_r2)
    pprint(search.best_params_)
    print(json.dumps(search.best_params_))

    optimized_model = get_regr(search.best_params_)
    optimized_model.fit(X, y)
    optimized_model.metadata = {'mae': avg_mae, 'r2': round(avg_r2, 2)}

    print("Saving %s" % MPDSExport.save_model(optimized_model, tag))
    print("Done in %1.2f sc" % (time.time() - starttime))
