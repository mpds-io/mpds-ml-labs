
import sys
import time

import httplib2
import numpy as np
from mpds_client import MPDSDataRetrieval, APIError

from prediction import prop_models
from struct_utils import detect_format, poscar_to_ase, refine, get_formula, sgn_to_crsystem
from cif_utils import cif_to_ase
from common import API_KEY, API_ENDPOINT, make_request


remote = httplib2.Http()
client = MPDSDataRetrieval(api_key=API_KEY, endpoint=API_ENDPOINT)

LABS_SERVER_ADDR = 'http://127.0.0.1:5000/predict'

ARITY = {1: 'unary', 2: 'binary', 3: 'ternary', 4: 'quaternary', 5: 'quinary'}


if __name__ == '__main__':

    try: sys.argv[1]
    except IndexError: sys.exit("Structure file must be given!")
    starttime = time.time()

    structure = open(sys.argv[1]).read()
    fmt = detect_format(structure)

    if fmt == 'cif':
        ase_obj, error = cif_to_ase(structure)
        if error:
            raise RuntimeError(error)

    elif fmt == 'poscar':
        ase_obj, error = poscar_to_ase(structure)
        if error:
            raise RuntimeError(error)

    else: raise RuntimeError('Provided data format is not supported')

    if 'disordered' in ase_obj.info:
        elements = sum(
            [item.keys() for item in ase_obj.info['disordered'].values()],
            []
        )
        elements = list(set(elements + [at.symbol for at in ase_obj]))
        tpl_query = {
            'elements': '-'.join(elements),
            'lattices': sgn_to_crsystem(ase_obj.info['spacegroup'].no)
        }
        if len(elements) in ARITY:
            tpl_query.update({'classes': ARITY[len(elements)]})

    else:
        ase_obj, error = refine(ase_obj)
        if error:
            raise RuntimeError(error)
        tpl_query = {
            'formulae': get_formula(ase_obj),
            'lattices': sgn_to_crsystem(ase_obj.info['spacegroup'].no)
        }

    answer = make_request(remote, LABS_SERVER_ADDR, {'structure': structure})
    if 'error' in answer:
        raise RuntimeError(answer['error'])

    for prop_id, pdata in prop_models.items():
        tpl_query.update({'props': pdata['name']})
        try:
            resp = client.get_dataframe(tpl_query)
        except APIError as e:
            prop_models[prop_id]['factual'] = None
            if e.code != 1:
                print("While checking against the MPDS an error %s occured" % e.code)
            continue

        resp['Value'] = resp['Value'].astype('float64') # to treat values out of bounds given as str
        resp = resp[resp['Units'] == pdata['units']]
        prop_models[prop_id]['factual'] = np.median(resp['Value'])

    for prop_id, pdata in answer['prediction'].items():
        print("{0:40} = {1:6}, factual {2:8} (MAE = {3:4}), {4}".format(
            prop_models[prop_id]['name'],
            'conductor' if pdata['value'] == 0 and prop_id == 'w' else pdata['value'],
            prop_models[prop_id]['factual'] or 'absent',
            pdata['mae'],
            prop_models[prop_id]['units']
        ))

    print("Done in %1.2f sc" % (time.time() - starttime))
