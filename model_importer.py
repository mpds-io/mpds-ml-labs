"""
Use to migrate and deploy models at the servers
with the different architecture, since the sklearn models
are not transferable
"""
import os
import sys
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from imblearn.over_sampling import RandomOverSampler

from mpds_client import MPDSExport
from mpds_ml_labs.prediction import estimate_regr_quality, estimate_clfr_quality


def get_regr(params={}):
    return RandomForestRegressor(**params)

def get_clfr(params={}):
    return RandomForestClassifier(**params)


results = []

DATA_DIR = '/data'

f = open(sys.argv[1], 'r')
final_values = json.loads(f.read())
f.close()

for key, value in final_values.items():
    print("*"*100)
    print("Importing model-%s" % key)

    if key == '0':
        white_data_file, black_data_file = os.path.join(DATA_DIR, value['white']), os.path.join(DATA_DIR, value['black'])
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

        error_percentage = estimate_clfr_quality(get_clfr(value['params']), X_resampled, y_resampled)
        print("Avg. error percentage: %.3f" % error_percentage)

        algo = get_clfr(value['params'])
        algo.fit(X_resampled, y_resampled)
        algo.metadata = {'error_percentage': error_percentage}

        export_file = MPDSExport.save_model(algo, 0)
        print("Saving %s" % export_file)
        results.append(export_file)

    else:
        data_file = os.path.join(DATA_DIR, value['file'])
        df = pd.read_pickle(data_file)
        X = np.array(df['Descriptor'].tolist())
        n_samples, n_x, n_y = X.shape
        X = X.reshape(n_samples, n_x * n_y)
        y = df['Avgvalue'].tolist()

        avg_mae, avg_r2 = estimate_regr_quality(get_regr(value['params']), X, y)
        print("Avg. MAE: %.2f; avg. R2 score: %.2f" % (avg_mae, avg_r2))

        algo = get_regr(value['params'])
        algo.fit(X, y)
        algo.metadata = {'mae': avg_mae, 'r2': round(avg_r2, 2)}

        export_file = MPDSExport.save_model(algo, key)
        print("Saving %s" % export_file)
        results.append(export_file)

for r in results:
    print(r)
