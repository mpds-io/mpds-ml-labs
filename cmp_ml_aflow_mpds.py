#!/usr/bin/env python2
"""
Given the folder with the POSCARs, this script employs two predictive ML models:
* AFLOW-ML PLMF (http://aflowlib.org/aflow-ml)
* MPDS ML (http://mpds.io/ml)
validating them using the experimental data from the core MPDS database
(a subscription is required), and outputs a CSV table for a detailed comparison
"""
from __future__ import division
import os
import sys
import time

import httplib2
import numpy as np
from ase.units import _Nav, _k
from mpds_client import MPDSDataRetrieval, APIError

from mpds_ml_labs.prediction import prop_models
from mpds_ml_labs.struct_utils import detect_format, poscar_to_ase, refine, get_formula, sgn_to_crsystem
from mpds_ml_labs.common import API_KEY, API_ENDPOINT, make_request
from mpds_ml_labs.aflowml_client import AFLOWmlAPI


RESULT_FILE = 'aflow_mpds_comparison_070219.csv'
LABS_SERVER_ADDR = 'https://labs.mpds.io/predict' # http://127.0.0.1:5000/predict
MPDS_AFLOW_CORR = {
    'z': 'ml_ael_bulk_modulus_vrh',
    'd': 'ml_agl_debye',
    't': 'ml_agl_thermal_expansion_300K',
    'x': 'ml_agl_heat_capacity_Cp_300K',
    'w': 'ml_egap'
}

def kbcell_to_jkmol(value, n_at_cell):
    return value * _k * _Nav / n_at_cell

assert not os.path.exists(RESULT_FILE)

try:
    given = sys.argv[1]
except IndexError:
    sys.exit("Structure file or folder with files must be given!")

tasks = []
if os.path.isdir(given):
    for filename in os.listdir(given):
        if not os.path.isfile(given + os.sep + filename):
            continue
        tasks.append(given + os.sep + filename)
else:
    tasks.append(given)

mpds_ml_remote = httplib2.Http()
mpds_api = MPDSDataRetrieval(api_key=API_KEY, endpoint=API_ENDPOINT, verbose=False)
aflowml = AFLOWmlAPI()
result_db = []

start_time = time.time()

for task in tasks:
    title = task.split(os.sep)[-1]
    structure = open(task).read()
    if detect_format(structure) != 'poscar':
        continue
    ase_obj, error = poscar_to_ase(structure)
    if error:
        continue
    if 'disordered' in ase_obj.info:
        continue
    ase_obj, error = refine(ase_obj)
    if error:
        continue
    formula, n_atoms_cell = get_formula(ase_obj), len(ase_obj)

    print("*"*20 + ("%s %s, %s" % (title, formula, n_atoms_cell)) + "*"*20)

    tpl_query = {
        'formulae': formula,
        'lattices': sgn_to_crsystem(ase_obj.info['spacegroup'].no)
    }

    results_conductor = 0
    try:
        outdf = mpds_api.get_dataframe(dict(classes='conductor', **tpl_query), fields={'P': [
            'sample.measurement[0].condition[0].name',
            'sample.measurement[0].condition[0].scalar',
            'sample.measurement[0].condition[0].units'
        ], 'S': [ # NB mockup, temperature to be released for S-entries soon
            lambda: 'Temperature',
            lambda: 300,
            lambda: 'K'
        ]}, columns=['Cname', 'Cvalue', 'Cunits'])
        to_drop = outdf[
            (outdf['Cname'] == 'Temperature') & (outdf['Cunits'] == 'K') & ((outdf['Cvalue'] < 200) | (outdf['Cvalue'] > 400))
        ]
        outdf.drop(to_drop.index, inplace=True)
        results_conductor = len(outdf)
    except APIError:
        pass

    time.sleep(1)

    mpds_output = make_request(mpds_ml_remote, LABS_SERVER_ADDR, {'structure': structure})
    if 'error' in mpds_output:
        continue

    aflow_output = aflowml.get_prediction(structure, 'plmf')

    for prop_id in MPDS_AFLOW_CORR.keys():
        try:
            outdf = mpds_api.get_dataframe(dict(props=prop_models[prop_id]['name'], **tpl_query), fields={'P': [
                'sample.material.chemical_formula',
                'sample.material.phase_id',
                'sample.measurement[0].property.scalar',
                'sample.measurement[0].property.units',
                'sample.measurement[0].condition[0].units',
                'sample.measurement[0].condition[0].name',
                'sample.measurement[0].condition[0].scalar'
            ]}, columns=['Compound', 'Phase', 'Value', 'Units', 'Cunits', 'Cname', 'Cvalue'])
        except APIError as e:
            prop_models[prop_id]['factual'] = None
            if e.code != 204: # NB standard code for the empty result
                print("While checking against the MPDS an error %s occured" % e.code)
            continue

        outdf = outdf[outdf['Units'] == prop_models[prop_id]['units']]
        outdf = outdf[
            (outdf['Value'] > prop_models[prop_id]['interval'][0]) & \
            (outdf['Value'] < prop_models[prop_id]['interval'][1])
        ]
        if prop_id not in ['m', 'd']:
            to_drop = outdf[
                (outdf['Cname'] == 'Temperature') & (outdf['Cunits'] == 'K') & ((outdf['Cvalue'] < 200) | (outdf['Cvalue'] > 400))
            ]
            outdf.drop(to_drop.index, inplace=True)
        if outdf.empty:
            prop_models[prop_id]['factual'] = None
            continue
        outdf['Value'] = outdf['Value'].astype('float64') # NB to treat values out of JSON bounds given as str
        prop_models[prop_id]['factual'] = np.median(outdf['Value'])

    # units conversion
    mpds_output['prediction']['t']['value'] /= 100000
    aflow_output[MPDS_AFLOW_CORR['x']] = kbcell_to_jkmol(aflow_output[MPDS_AFLOW_CORR['x']], n_atoms_cell)

    # remark on conductivity
    results_insulator = prop_models['w']['factual'] and np.isfinite(prop_models['w']['factual'])
    if results_insulator and results_conductor:
        remark = 'Semiconductor'
    elif results_insulator:
        remark = 'Insulator'
    elif results_conductor:
        remark = 'Conductor'
    else:
        remark = 'Unknown'

    result_db.append([
        title, formula, n_atoms_cell,
        prop_models['z']['name'], prop_models['z']['factual'], aflow_output[MPDS_AFLOW_CORR['z']], mpds_output['prediction']['z']['value'], '', '',
        prop_models['d']['name'], prop_models['d']['factual'], aflow_output[MPDS_AFLOW_CORR['d']], mpds_output['prediction']['d']['value'], '', '',
        prop_models['t']['name'], prop_models['t']['factual'], aflow_output[MPDS_AFLOW_CORR['t']], mpds_output['prediction']['t']['value'], '', '',
        prop_models['x']['name'], prop_models['x']['factual'], aflow_output[MPDS_AFLOW_CORR['x']], mpds_output['prediction']['x']['value'], '', '',
        prop_models['w']['name'], prop_models['w']['factual'], aflow_output[MPDS_AFLOW_CORR['w']], mpds_output['prediction']['w']['value'], '', '',
        remark
    ])

print("Done in %1.2f sc" % (time.time() - start_time))

f_result = open(RESULT_FILE, "w")
for row in result_db:
    f_result.write(",".join([str(item) for item in row]) + "\n")
f_result.close()
