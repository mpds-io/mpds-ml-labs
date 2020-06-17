
import time
import httplib2
import ujson as json

from cif_utils import cif_to_ase
from common import make_request


remote = httplib2.Http()

LABS_SERVER_ADDR = 'http://127.0.0.1:5000/design'

# NB. mind prediction_ranges.prediction_ranges
sample = {
    'z': [200, 265],
    'y': [-325, -250],
    'x': [11, 28],
    'k': [150, 225],
    'w': [1, 3],
    'm': [2000, 2700],
    'd': [175, 1100],
    't': [-0.5, 3],
    'i': [6, 16],
    'o': [0, 40]
}

if __name__ == '__main__':
    starttime = time.time()

    answer = make_request(remote, LABS_SERVER_ADDR, {'numerics': json.dumps(sample)})
    if 'error' in answer:
        raise RuntimeError(answer['error'])

    #print(answer['vis_cif'])
    ase_obj, error = cif_to_ase(answer['vis_cif'])
    if error:
        raise RuntimeError(error)

    print("Done in %1.2f sc" % (time.time() - starttime))
