
import os
from ConfigParser import ConfigParser


DATA_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '../data'))
config = ConfigParser()
config_path = os.path.join(DATA_PATH, 'settings.ini')

if os.path.exists(config_path):
    config.read(config_path)

    SERVE_UI = config.get('mpds_ml_labs', 'serve_ui')
    ML_MODELS = config.get('mpds_ml_labs', 'ml_models') or ''
    API_KEY = config.get('mpds_ml_labs', 'api_key')
    API_ENDPOINT = config.get('mpds_ml_labs', 'api_endpoint')

    ML_MODELS = [
        path.strip() for path in filter(None, ML_MODELS.split())
    ]

else:
    SERVE_UI = True
    ML_MODELS = []
    API_KEY = None
    API_ENDPOINT = None
