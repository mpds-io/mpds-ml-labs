
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
    "n_estimators": list(range(200, 1001, 20)),
    "max_features": list(range(10, 30)),
    "max_depth": [None, 25, 37, 50, 75, 100, 125, 150],
    "min_samples_split": list(range(2, 15)),
    "min_samples_leaf": list(range(1, 9)),
    "bootstrap": [False],
    "criterion": ['mae'],
    "n_jobs": [-1]
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
    n_samples, n_x, n_y = X.shape
    X = X.reshape(n_samples, n_x * n_y)
    y = df['Avgvalue'].tolist()

    starttime = time.time()

    search = RandomizedSearchCV(get_regr(), param_distributions=param_dist, n_iter=2500, cv=2, verbose=3)
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
