
import os
from ConfigParser import ConfigParser
from urllib import urlencode

import ujson as json
import pg8000


DATA_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '../data'))
config = ConfigParser()
config_path = os.path.join(DATA_PATH, 'settings.ini')

if os.path.exists(config_path):
    config.read(config_path)

    SERVE_UI = config.getboolean('mpds_ml_labs', 'serve_ui')
    ML_MODELS = config.get('mpds_ml_labs', 'ml_models') or ''
    API_KEY = config.get('mpds_ml_labs', 'api_key')
    API_ENDPOINT = config.get('mpds_ml_labs', 'api_endpoint')
    ELS_ENDPOINT = config.get('mpds_ml_labs', 'els_endpoint')

    ML_MODELS = [
        path.strip() for path in filter(None, ML_MODELS.split())
    ]

    KNN_TABLE = config.get('db', 'table')

else:
    SERVE_UI = True
    ML_MODELS = []
    API_KEY = None
    API_ENDPOINT = None
    ELS_ENDPOINT = None

    KNN_TABLE = None


def connect_database():

    assert KNN_TABLE

    connection = pg8000.connect(
        user=config.get('db', 'user'),
        password=config.get('db', 'password'),
        database=config.get('db', 'database'),
        host=config.get('db', 'host'),
        port=config.getint('db', 'port')
    )
    cursor = connection.cursor()

    return cursor, connection


def make_request(req, address, data={}, httpverb='POST', headers={}):

    address += '?' + urlencode(data)

    if httpverb == 'GET':
        response, content = req.request(address, httpverb, headers=headers)

    else:
        headers.update({'Content-type': 'application/x-www-form-urlencoded'})
        response, content = req.request(address, httpverb, headers=headers, body=urlencode(data))

    return json.loads(content)
