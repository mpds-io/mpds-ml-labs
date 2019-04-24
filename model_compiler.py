
import os, sys

import treelite.gallery.sklearn
from mpds_ml_labs.prediction import load_ml_models
#from mpds_ml_labs.common import ML_MODELS


assert os.path.exists(sys.argv[1])

#active_ml_models = load_ml_models(ML_MODELS)
active_ml_models = load_ml_models([sys.argv[1]])

#mod_path = '/data/models_cmpld'
mod_path = './'
mod_basename = '_cmpld.so'

for prop_id in active_ml_models:
    assert not os.path.exists(mod_path + os.sep + prop_id + mod_basename)

for prop_id in active_ml_models:
    print("Compiling model %s" % prop_id)
    i_model = treelite.gallery.sklearn.import_model(active_ml_models[prop_id])
    i_model.export_lib(toolchain='clang', libpath=mod_path + os.sep + prop_id + mod_basename, verbose=True) # , params={'parallel_comp': 8}
    print("Done with %s" % prop_id)
