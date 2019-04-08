
from __future__ import division
import time

from progressbar import ProgressBar
import numpy as np
import pandas as pd
from mpds_client import MPDSDataRetrieval, MPDSExport

from mpds_ml_labs.prediction import get_ordered_descriptor, get_aligned_descriptor
from mpds_ml_labs.struct_utils import json_to_ase
from mpds_ml_labs.common import API_KEY, API_ENDPOINT


stepwise_conditions = ("1920-1970", "1971-1990", "1991-2000", "2001-2008", "2009-2020")

client = MPDSDataRetrieval(api_key=API_KEY, endpoint=API_ENDPOINT)

starttime = time.time()

cond_candidates_phases, ins_candidates_phases = [], []

for stepwise_condition in stepwise_conditions:
    for phase_id in client.get_data(
        {"classes": "conductor", "years": stepwise_condition},
        fields={'P': ['sample.material.phase_id'], 'S': ['phase_id']},
    ):
        if not phase_id[0]:
            continue
        cond_candidates_phases.append(phase_id[0])

for stepwise_condition in stepwise_conditions:
    for phase_id in client.get_data(
        {"props": "band gap", "years": stepwise_condition},
        fields={'P': ['sample.material.phase_id']},
    ):
        if not phase_id[0]:
            continue
        ins_candidates_phases.append(phase_id[0])

cond_candidates_phases, ins_candidates_phases = set(cond_candidates_phases), set(ins_candidates_phases)

print("Conductor candidate phases: %s" % len(cond_candidates_phases))
print("Insulator candidate phases: %s" % len(ins_candidates_phases))

cond_phases = cond_candidates_phases - ins_candidates_phases
ins_phases = ins_candidates_phases - cond_candidates_phases
semi_phases = cond_candidates_phases & ins_candidates_phases

print("True conductor phases: %s" % len(cond_phases))
print("True insulator phases: %s" % len(ins_phases))
print("Semiconductor phases: %s" % len(semi_phases))

'''rt_ins_phases = set()

for row in client.get_data(
    {"props": "physical hierarchy"},
    phases=semi_phases,
    fields={'P': [
        'sample.material.phase_id',
        'sample.measurement[0].condition[0].units',
        'sample.measurement[0].condition[0].name',
        'sample.measurement[0].condition[0].scalar'
    ]}
):
    if row[1] != 'K' or row[2] != 'Temperature':
        continue
    if row[3] < 200 or row[3] > 400:
        continue
    rt_ins_phases.add(row[0])

print("Insulating phases at near RT: %s" % len(rt_ins_phases))'''

def get_crystal_descriptors(phases, tag):

    data_by_phases = {}

    for stepwise_condition in stepwise_conditions:
        print("Stepwise iteration")
        for item in ProgressBar()(client.get_data(
            {
                "props": "atomic structure",
                "years": stepwise_condition
            },
            fields={'S':['phase_id', 'entry', 'occs_noneq', 'cell_abc', 'sg_n', 'basis_noneq', 'els_noneq']},
            phases=phases
        )):
            ase_obj, error = json_to_ase(item)
            if error: continue

            if 'disordered' in ase_obj.info:
                descriptor, error = get_ordered_descriptor(ase_obj)
                if error: continue

            else:
                descriptor, error = get_aligned_descriptor(ase_obj)
                if error: continue

            if item[0] in data_by_phases:
                left_len, right_len = len(data_by_phases[item[0]][0]), len(descriptor[0])

                if left_len != right_len: # align length
                    data_by_phases[item[0]] = data_by_phases[item[0]][:, :min(left_len, right_len)]
                    descriptor =                           descriptor[:, :min(left_len, right_len)]

                data_by_phases[item[0]] = (data_by_phases[item[0]] + descriptor)/2
            else:
                data_by_phases[item[0]] = descriptor

    min_len = min([len(x[0]) for x in data_by_phases.values()])
    for phase_id in data_by_phases.keys():
        if len(data_by_phases[phase_id][0]) > min_len:
            data_by_phases[phase_id] = data_by_phases[phase_id][:, :min_len]
        data_by_phases[phase_id] = data_by_phases[phase_id].flatten()

    print("Current descriptor length: %d" % min_len)

    structs = pd.DataFrame(list(data_by_phases.items()), columns=['Phase', 'Descriptor'])
    export = MPDSExport.save_df(structs, tag)
    print("Saved %s" % export)

get_crystal_descriptors(cond_phases, 0)
#get_crystal_descriptors(rt_ins_phases | ins_phases, 1)
get_crystal_descriptors(ins_phases, 1)

print("Done in %1.2f sc" % (time.time() - starttime))
