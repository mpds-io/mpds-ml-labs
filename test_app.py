"""
Could be tested also like:
curl -XPOST http://localhost:8523 -d "structure=123"
"""
import os

from urllib import urlencode

import httplib2
import ujson as json


req = httplib2.Http()

def make_request(address, data={}, httpverb='POST', headers={}):

    address += '?' + urlencode(data)

    if httpverb == 'GET':
        response, content = req.request(address, httpverb, headers=headers)

    else:
        headers.update({'Content-type': 'application/x-www-form-urlencoded'})
        response, content = req.request(address, httpverb, headers=headers, body=urlencode(data))

    return json.loads(content)

if __name__ == '__main__':

    for root, dirs, files in os.walk('/home/Evgeny/data/poscar_examples'):
        for f in files:
            tname = os.path.join(root, f)
            subject = open(tname).read()
            answer = make_request('http://127.0.0.1:8523', {'structure': subject})

            if 'error' in answer:
                print tname, 'ERROR', answer['error']
            else:
                print tname, answer['prediction']
