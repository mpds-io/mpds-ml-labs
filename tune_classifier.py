
from __future__ import division
import os, sys
import time
import json
from pprint import pprint

import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from mpds_client import MPDSExport

from mpds_ml_labs.prediction import prop_models, estimate_clfr_quality


def get_clfr(params={}):
    return RandomForestClassifier(**params)

param_dist = {
    "n_estimators": list(range(25, 501, 25)) + [750, 1000],
    "max_features": list(range(10, 91, 4)) + [100, 200],
    "max_depth": [None, 10, 25, 50, 100],
    "min_samples_split": [2, 4, 10, 16],
    "min_samples_leaf": [1, 3, 5, 7, 14],
    "bootstrap": [True, False],
    "n_jobs": [-1]
}

if __name__ == "__main__":
    white_data_file, black_data_file = sys.argv[1], sys.argv[2]
    if not os.path.exists(white_data_file) or not os.path.exists(black_data_file):
        raise RuntimeError

    print("Data file (0): %s" % white_data_file)
    print("Data file (1): %s" % black_data_file)

    white_df, black_df = pd.read_pickle(white_data_file), pd.read_pickle(black_data_file)
    white_df['Class'] = 0
    black_df['Class'] = 1
    all_df = pd.concat([white_df, black_df])
    X = all_df['Descriptor'].tolist()
    y = all_df['Class'].tolist()

    min_x_len = min([len(j) for j in X])
    for n in range(len(X)):
        if len(X[n]) > min_x_len:
            X[n] = X[n][:min_x_len]

    X = np.array(X, dtype=float)

    ros = RandomOverSampler()
    X_resampled, y_resampled = ros.fit_sample(X, y)

    print("White new len: %s" % (len([m for m in y_resampled if m == 0])))
    print("Black new len: %s" % (len([m for m in y_resampled if m == 1])))

    starttime = time.time()

    search = RandomizedSearchCV(get_clfr(), param_distributions=param_dist, n_iter=2500, cv=2, verbose=3)
    search.fit(X_resampled, y_resampled)
    error_percentage = estimate_clfr_quality(get_clfr(search.best_params_), X_resampled, y_resampled)

    print("Avg. error percentage: %.3f" % error_percentage)
    pprint(search.best_params_)
    print(json.dumps(search.best_params_))

    optimized_model = get_clfr(search.best_params_)
    optimized_model.fit(X_resampled, y_resampled)
    optimized_model.metadata = {'error_percentage': error_percentage}

    print("Saving %s" % MPDSExport.save_model(optimized_model, 0))
    print("Done in %1.2f sc" % (time.time() - starttime))
