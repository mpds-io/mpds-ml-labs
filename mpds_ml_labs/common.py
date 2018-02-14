
import os
from ConfigParser import ConfigParser


DATA_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '../data'))
config = ConfigParser()
config_path = os.path.join(DATA_PATH, 'settings.ini')

if os.path.exists(config_path):
    config.read(config_path)
    SERVE_UI = config.get('mpds_ml_labs', 'serve_ui')
    ML_MODELS = [path.strip() for path in filter(
        None,
        config.get('mpds_ml_labs', 'ml_models').split()
    )]
else:
    SERVE_UI = True
    ML_MODELS = []
