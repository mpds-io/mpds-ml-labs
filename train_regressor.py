
from __future__ import division
import os, sys
import time
from progressbar import ProgressBar

import numpy as np
import pandas as pd
from mpds_client import MPDSDataRetrieval, MPDSExport

from mpds_ml_labs.prediction import prop_models, get_aligned_descriptor, get_ordered_descriptor, get_regr, estimate_regr_quality
from mpds_ml_labs.struct_utils import json_to_ase
from mpds_ml_labs.common import API_KEY, API_ENDPOINT


def mpds_get_data(api_client, prop_id, descriptor_kappa):
    """
    Fetch, massage, and save dataframe from the MPDS
    NB currently pressure is not taken into account!
    """
    print("Getting %s with descriptor kappa = %s" % (prop_models[prop_id]['name'], descriptor_kappa))
    starttime = time.time()

    props = api_client.get_dataframe(
        {"props": prop_models[prop_id]['name']},
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
    props['Value'] = props['Value'].astype('float64') # to treat values out of bounds given as str
    props = props[np.isfinite(props['Phase'])]
    props = props[props['Units'] == prop_models[prop_id]['units']]
    props = props[
        (props['Value'] > prop_models[prop_id]['interval'][0]) & \
        (props['Value'] < prop_models[prop_id]['interval'][1])
    ]

    if prop_id not in ['m', 'd']:
        to_drop = props[
            (props['Cname'] == 'Temperature') & (props['Cunits'] == 'K') & ((props['Cvalue'] < 200) | (props['Cvalue'] > 400))
        ]
        print("Rows to neglect by temperature: %s" % len(to_drop))
        props.drop(to_drop.index, inplace=True)

    if prop_id == 't':
        props['Value'] *= 100000 # scaling 10**5

    phases_compounds = dict(zip(props['Phase'], props['Compound'])) # keep the mapping for future
    avgprops = props.groupby('Phase')['Value'].mean().to_frame().reset_index().rename(columns={'Value': 'Avgvalue'})
    phases = np.unique(avgprops['Phase'].astype(int)).tolist()

    print("Got %s distinct crystalline phases" % len(phases))

    data_by_phases = {}

    print("Computing descriptors...")
    pbar = ProgressBar()
    for item in pbar(api_client.get_data(
        {"props": "atomic structure"},
        fields={'S':['phase_id', 'entry', 'occs_noneq', 'cell_abc', 'sg_n', 'basis_noneq', 'els_noneq']},
        phases=phases
    )):
        ase_obj, error = json_to_ase(item)
        if error: continue

        if 'disordered' in ase_obj.info:
            descriptor, error = get_ordered_descriptor(ase_obj, kappa=descriptor_kappa)
            if error: continue

        else:
            descriptor, error = get_aligned_descriptor(ase_obj, kappa=descriptor_kappa)
            if error: continue

        if item[0] in data_by_phases:
            left_len, right_len = len(data_by_phases[item[0]][0]), len(descriptor[0])

            if left_len != right_len: # align length
                data_by_phases[item[0]] = data_by_phases[item[0]][:, :min(left_len, right_len)]
                descriptor = descriptor[:, :min(left_len, right_len)]

            data_by_phases[item[0]] = (data_by_phases[item[0]] + descriptor)/2

        else:
            data_by_phases[item[0]] = descriptor

    min_len = min([len(x[0]) for x in data_by_phases.values()])
    for phase_id in data_by_phases.keys():
        if len(data_by_phases[phase_id][0]) > min_len:
            data_by_phases[phase_id] = data_by_phases[phase_id][:, :min_len]

    print("Current descriptor length: %d" % min_len)

    structs = pd.DataFrame(list(data_by_phases.items()), columns=['Phase', 'Descriptor'])
    struct_props = structs.merge(avgprops, how='outer', on='Phase')
    struct_props = struct_props[struct_props['Descriptor'].notnull()]
    struct_props['Phase'] = struct_props['Phase'].map(phases_compounds)
    struct_props.rename(columns={'Phase': 'Compound'}, inplace=True)

    print("Done %s rows in %1.2f sc" % (len(struct_props), time.time() - starttime))

    struct_props.export_file = MPDSExport.save_df(struct_props, prop_id)
    print("Saving %s" % struct_props.export_file)

    return struct_props


def tune_model(data_file):
    """
    Load saved data and perform a simple regressor parameter tuning
    """
    basename = data_file.split(os.sep)[-1]
    if basename.startswith('df') and basename[3:4] == '_' and basename[2:3] in prop_models:
        tag = basename[2:3]
        print("Detected property %s" % prop_models[tag]['name'])
    else:
        tag = None
        print("No property name detected")

    df = pd.read_pickle(data_file)

    X = np.array(df['Descriptor'].tolist())
    n_samples, n_x, n_y = X.shape
    X = X.reshape(n_samples, n_x * n_y)
    y = df['Avgvalue'].tolist()

    results = []
    for parameter_a in range(20, 501, 20):
        avg_mae, avg_r2 = estimate_regr_quality(get_regr(a=parameter_a), X, y)
        results.append([parameter_a, avg_mae, avg_r2])
        print("%s\t\t\t%s\t\t\t%s" % (parameter_a, avg_mae, avg_r2))
    results.sort(key=lambda x: (-x[1], x[2]))

    print("Best result:", results[-1])
    parameter_a = results[-1][0]

    results = []
    for parameter_b in range(10, 101, 2):
        avg_mae, avg_r2 = estimate_regr_quality(get_regr(a=parameter_a, b=parameter_b), X, y)
        results.append([parameter_b, avg_mae, avg_r2])
        print("%s\t\t\t%s\t\t\t%s" % (parameter_b, avg_mae, avg_r2))
    results.sort(key=lambda x: (-x[1], x[2]))

    print("Best result:", results[-1])
    parameter_b, avg_mae, avg_r2 = results[-1]

    print("a = %s b = %s" % (parameter_a, parameter_b))

    regr = get_regr(a=parameter_a, b=parameter_b)
    regr.fit(X, y)
    regr.metadata = {'mae': avg_mae, 'r2': round(avg_r2, 2)}

    if tag:
        export_file = MPDSExport.save_model(regr, tag)
        print("Saving %s" % export_file)


if __name__ == "__main__":
    try:
        arg = sys.argv[1]
    except IndexError:
        sys.exit(
    "What to do?\n"
    "Please, provide either a *prop_id* letter (%s) for a property data to be downloaded and fitted,\n"
    "or a data *filename* for tuning the model." % ", ".join(prop_models.keys())
        )
    try:
        descriptor_kappa = int(sys.argv[2])
    except:
        descriptor_kappa = None

    if arg in prop_models.keys():

        api_client = MPDSDataRetrieval(api_key=API_KEY, endpoint=API_ENDPOINT)

        struct_props = mpds_get_data(api_client, arg, descriptor_kappa)

        X = np.array(struct_props['Descriptor'].tolist())
        n_samples, n_x, n_y = X.shape
        X = X.reshape(n_samples, n_x * n_y)
        y = struct_props['Avgvalue'].tolist()

        avg_mae, avg_r2 = estimate_regr_quality(get_regr(), X, y)

        print("Avg. MAE: %.2f" % avg_mae)
        print("Avg. R2 score: %.2f" % avg_r2)

        tune_model(struct_props.export_file)

    elif os.path.exists(arg):
        tune_model(arg)

    else: raise RuntimeError("Unrecognized argument: %s" % arg)
