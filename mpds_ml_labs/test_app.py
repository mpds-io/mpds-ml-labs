
import sys
from urllib import urlencode

import httplib2
import ujson as json
import numpy as np

from mpds_client import MPDSDataRetrieval, APIError

from prediction import prop_semantics
from struct_utils import detect_format, poscar_to_ase, symmetrize, get_formula, sgn_to_crsystem
from cif_utils import cif_to_ase


req = httplib2.Http()
client = MPDSDataRetrieval()

def make_request(address, data={}, httpverb='POST', headers={}):

    address += '?' + urlencode(data)

    if httpverb == 'GET':
        response, content = req.request(address, httpverb, headers=headers)

    else:
        headers.update({'Content-type': 'application/x-www-form-urlencoded'})
        response, content = req.request(address, httpverb, headers=headers, body=urlencode(data))

    return json.loads(content)

if __name__ == '__main__':

    try: sys.argv[1]
    except IndexError: sys.exit("Structure file must be given!")

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

    else:
        raise RuntimeError('Provided data format is not supported')

    ase_obj, error = symmetrize(ase_obj)
    if error:
        raise RuntimeError(error)

    answer = make_request('http://127.0.0.1:5000/predict', {'structure': structure})
    if 'error' in answer:
        raise RuntimeError(answer['error'])

    formulae_categ, lattices_categ = get_formula(ase_obj), sgn_to_crsystem(ase_obj.info['spacegroup'].no)
    for prop_id, pdata in prop_semantics.items():
        try:
            resp = client.get_dataframe({
                'formulae': formulae_categ,
                'lattices': lattices_categ,
                'props': pdata['name']
            })
        except APIError as e:
            prop_semantics[prop_id]['factual'] = None
            if e.code == 1:
                continue
            else:
                raise

        resp['Value'] = resp['Value'].astype('float64') # to treat values out of bounds given as str
        resp = resp[resp['Units'] == pdata['units']]
        prop_semantics[prop_id]['factual'] = np.median(resp['Value'])

    for prop_id, pdata in answer['prediction'].items():
        print("{0:40} = {1:6}, factual {2:8} (MAE = {3:4}), {4}".format(
            prop_semantics[prop_id]['name'],
            pdata['value'],
            prop_semantics[prop_id]['factual'] or 'absent',
            pdata['mae'],
            prop_semantics[prop_id]['units']
        ))
