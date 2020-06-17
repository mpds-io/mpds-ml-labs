
import time
import httplib2
import ujson as json

from cif_utils import cif_to_ase
from common import make_request


remote = httplib2.Http()

LABS_SERVER_ADDR = 'http://127.0.0.1:5000/design'

# NB. mind prediction_ranges.prediction_ranges
sample = {
    'z': [39, 284],
    'y': [-279, -27],
    'x': [13, 24],
    'k': [-92, 245],
    'w': [1.5, 7.7],
    'm': [72, 2594],
    'd': [159, 999],
    't': [1.1, 43.0],
    'i': [-17, 12],
    'o': [7, 106]
}

if __name__ == '__main__':
    starttime = time.time()

    answer = make_request(remote, LABS_SERVER_ADDR, {'numerics': json.dumps(sample)})
    if 'error' in answer:
        raise RuntimeError(answer['error'])

    _, error = cif_to_ase(answer['vis_cif'])
    assert not error, error
    print(answer['vis_cif'][:1000])

    print("Done in %1.2f sc" % (time.time() - starttime))
