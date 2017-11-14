
from __future__ import division
import os, sys
import time
import random

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from mpds_client import MPDSDataRetrieval, MPDSExport

from prediction import get_descriptor, human_names


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


def estimate_quality(algo, args, values, attempts=40, nsamples=40):
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


def mpds_get_data(prop_id):
    """
    NB
    currently pressure is not taken into account,
    however must be
    """
    print("Getting %s" % human_names[prop_id]['name'])
    starttime = time.time()

    client = MPDSDataRetrieval()

    props = client.get_dataframe(
        {"props": human_names[prop_id]['name']},
        fields={'P': [
            'sample.material.chemical_formula',
            'sample.material.phase_id',
            'sample.measurement[0].property.scalar',
            'sample.measurement[0].property.units',
            'sample.measurement[0].condition[0].units',
            'sample.measurement[0].condition[0].name',
            'sample.measurement[0].condition[0].scalar'
        ]},
        columns=['Compound', 'Phase', 'Value', 'Units', 'Cunits', 'Cname', 'Cvalue']
    )
    props = props[np.isfinite(props['Phase'])]
    props = props[props['Units'] == human_names[prop_id]['units']]
    if prop_id == 'w': # to filter several abnormal values
        props = props[(props['Value'] > 0) & (props['Value'] < 20)]
    # TODO

    to_drop = props[(props['Cname'] == 'Temperature') & (props['Cunits'] == 'K') & (props['Cvalue'] > 500)]
    print("Rows with high-T values to drop:", len(to_drop))
    props.drop(to_drop.index, inplace=True)

    phases_compounds = dict(zip(props['Phase'], props['Compound'])) # keep the mapping for future
    avgprops = props.groupby('Phase')['Value'].mean().to_frame().reset_index().rename(columns={'Value': 'Avgvalue'})
    phases = avgprops['Phase'].astype(int).tolist()

    print("Got %s distinct crystalline phases" % len(phases))

    min_descriptor_len, max_descriptor_len = 120, 1200
    data_by_phases = {}

    for item in client.get_data(
        {"props": "atomic structure"},
        fields={'S':['phase_id', 'entry', 'chemical_formula', 'cell_abc', 'sg_n', 'setting', 'basis_noneq', 'els_noneq']},
        phases=phases
    ):
        crystal = MPDSDataRetrieval.compile_crystal(item, 'ase')
        if not crystal: continue
        descriptor = get_descriptor(crystal)

        if len(descriptor) < min_descriptor_len:
            print("Entry %s: not enough atoms, cannot get reliable desciptor" % item[1])
            continue

        if len(descriptor) < max_descriptor_len:
            max_descriptor_len = len(descriptor)

        if item[0] in data_by_phases:
            left_len, right_len = len(data_by_phases[item[0]]), len(descriptor)

            if left_len != right_len: # align length
                data_by_phases[item[0]] = data_by_phases[item[0]][:min(left_len, right_len)]
                descriptor = descriptor[:min(left_len, right_len)]

            data_by_phases[item[0]] = (data_by_phases[item[0]] + descriptor)/2

        else:
            data_by_phases[item[0]] = descriptor

    for phase_id in data_by_phases.keys():
        data_by_phases[phase_id] = data_by_phases[phase_id][:max_descriptor_len]

    print("Current descriptor length: %d" % max_descriptor_len)

    structs = pd.DataFrame(list(data_by_phases.items()), columns=['Phase', 'Descriptor'])
    struct_props = structs.merge(avgprops, how='outer', on='Phase')
    struct_props = struct_props[struct_props['Descriptor'].notnull()]
    struct_props['Phase'] = struct_props['Phase'].map(phases_compounds)
    struct_props.rename(columns={'Phase': 'Compound'}, inplace=True)

    print("Done %s rows in %1.2f sc" % (len(struct_props), time.time() - starttime))

    export_file = MPDSExport.export_df(struct_props, prop_id)
    print("Saving %s" % export_file)

    return struct_props


if __name__ == "__main__":
    try:
        arg = sys.argv[1]
    except IndexError:
        raise RuntimeError("\n\nWhat to do?\n\nPlease, provide a *prop_id* letter OR data *filename*")

    if arg in human_names.keys():

        # getting the data from scratch by prop_id
        struct_props = mpds_get_data(arg)

        X = np.array(struct_props['Descriptor'].tolist())
        y = struct_props['Avgvalue'].tolist()

        avg_mae, avg_r2 = estimate_quality(get_regr(), X, y)
        print("Avg. MAE: %.2f" % avg_mae)
        print("Avg. R2 score: %.2f" % avg_r2)

    elif os.path.exists(arg):

        # loading saved data
        basename = arg.split(os.sep)[-1]
        if basename.startswith('df') and basename[3:4] == '_' and basename[2:3] in human_names:
            tag = basename[2:3]
            print("Detected property %s" % human_names[tag]['name'])
        else:
            tag = None
            print("No property name detected")

        df = pd.read_pickle(arg)

        X = np.array(df['Descriptor'].tolist())
        y = df['Avgvalue'].tolist()

        # simple regressor parameter tuning
        results = []
        for a in range(60, 501, 20):
            avg_mae, avg_r2 = estimate_quality(get_regr(a=a), X, y)
            results.append([a, avg_mae, avg_r2])
            print("%s\t\t\t%s\t\t\t%s" % (a, avg_mae, avg_r2))
        results.sort(key=lambda x: (-x[1], x[2]))

        print("Best result:", results[-1])
        a = results[-1][0]

        results = []
        for b in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            avg_mae, avg_r2 = estimate_quality(get_regr(a=a, b=b), X, y)
            results.append([b, avg_mae, avg_r2])
            print("%s\t\t\t%s\t\t\t%s" % (b, avg_mae, avg_r2))
        results.sort(key=lambda x: (-x[1], x[2]))

        print("Best result:", results[-1])
        b = results[-1][0]

        print("a = %s b = %s" % (a, b))

        regr = get_regr(a=a, b=b)
        regr.fit(X, y)
        regr.metadata = {'mae': avg_mae, 'r2': round(avg_r2, 2)}

        if tag:
            export_file = MPDSExport.export_model(regr, tag)
            print("Saving %s" % export_file)

    else: raise RuntimeError("Unrecognized argument: %s" % arg)
