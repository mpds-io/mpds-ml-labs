Data-driven predictions: from crystal structure to physical properties and vice versa
======

[![DOI](https://zenodo.org/badge/110734326.svg)](https://zenodo.org/badge/latestdoi/110734326)

![Materials simulations ab datum](https://raw.githubusercontent.com/mpds-io/mpds-ml-labs/master/crystallographer_mpds_cc_by_40.png "Materials simulation ab datum")


Live demos
------

[mpds.io/ml](https://mpds.io/ml) and [mpds.io/materials-design](https://mpds.io/materials-design)


Rationale
------

This is the proof of concept, how a relatively unsophisticated statistical model (namely, _random forest regressor_) trained on the large MPDS dataset predicts a set of physical properties from the only crystalline structure. Similarly to _ab initio_, this method could be called _ab datum_. (Note however that the simulation of physical properties with a comparable precision normally takes days, weeks or even months, whereas the present method takes less than a second!) A crystal structure in either CIF or POSCAR format is required. The following physical properties are predicted:

- isothermal bulk modulus
- enthalpy of formation
- heat capacity at constant pressure
- melting temperature
- Debye temperature
- Seebeck coefficient
- linear thermal expansion coefficient
- band gap (or its absense, _i.e._ whether a crystal is conductor or insulator)

Further, a reverse task of predicting the possible crystalline structure from a set of given properties is solved. The suitable chemical elements are found, and the resulted structure is generated (if possible) based on the available MPDS prototypes.


Installation
------

```shell
git clone REPO_ADDR
virtualenv --system-site-packages REPO_FOLDER
cd REPO_FOLDER
. bin/activate
pip install -r requirements.txt
```

Currently only *Python 2* is supported (*Python 3* support is almost there).


Preparation
------

The model is trained on the MPDS data using the MPDS API and the scripts `train_regressor.py` and `train_classifier.py`. Some subset of the full MPDS data is opened and possible to obtain via MPDS API [for free](https://mpds.io/open-data-api).


Architecture and usage
------

Can be used either as a standalone command-line application or as a client-server application. In the latter case, the client and the server communicate over HTTP, and any client able to execute HTTP requests is supported, be it a `curl` command-line client or rich web-browser user interface. For example, the simple HTML5 apps `props.html` and `design.html` are supplied in the `webassets` folder. Server part is a Flask app:

```python
python mpds_ml_labs/app.py
```

Web-browser user interface is then available under `http://localhost:5000`. By default, to serve the requests the development Flask server is used. Therefore an _AS-IS_ deployment in an online environment without the suitable WSGI container is **highly discouraged**. For the production environments under the high load it is recommended to use something like [TensorFlow Serving](https://www.tensorflow.org/serving).


Used descriptor and model details
------

The term _descriptor_ stands for the compact information-rich representation, allowing the convenient mathematical treatment of the encoded complex data (_i.e._ crystalline structure). Any crystalline structure is populated to a certain relatively big fixed volume of minimum one cubic nanometer. Then the descriptor is constructed using the periodic numbers of atoms and the lengths of their radius-vectors. The details are in the file `mpds_ml_labs/prediction.py`.

As a machine-learning model an ensemble of decision trees ([random forest regressor](http://scikit-learn.org/stable/modules/ensemble.html)) is used, as implemented in [scikit-learn](http://scikit-learn.org) Python machine-learning toolkit. The whole MPDS dataset can be used for training. In order to estimate the prediction quality of the _regressor_ model, the _mean absolute error_ and _R2 coefficient of determination_ is saved. In order to estimate the prediction quality of the binary _classifier_ model, the _fraction incorrect_ (_i.e._ the _error percentage_) is saved. The evaluation process is repeated at least 30 times to achieve a statistical reliability.

For generating the crystal structure from the physical properties, see `mpds_ml_labs/test_design.py`.


API
------

At the local server:

```shell
curl -XPOST http://localhost:5000/predict -d "structure=data_in_CIF_or_POSCAR"
curl -XPOST http://localhost:5000/design -d "numerics=ranges_of_values_of_the_8_properties_in_JSON"
```

At the demonstration MPDS server (may be switched off):

```shell
curl -XPOST https://labs.mpds.io/predict -d "structure=data_in_CIF_or_POSCAR"
curl -XPOST https://labs.mpds.io/design -d "numerics=ranges_of_values_of_the_8_properties_in_JSON"
```


Credits
------

This project is built on top of the open-source scientific software, such as:

- [scikit-learn](http://scikit-learn.org)
- [pandas](https://pandas.pydata.org)
- [ASE](https://wiki.fysik.dtu.dk/ase)
- [pycodcif](http://wiki.crystallography.net/cod-tools/CIF-parser)
- [spglib](https://atztogo.github.io/spglib)
- [cifplayer](http://tilde-lab.github.io/player.html)
- [MPDS API client](http://developer.mpds.io)


License
------

- The client and the server code: *LGPL-2.1+*
- The machine-learning MPDS data generated as presented here: *CC BY 4.0*
- The [open part](https://mpds.io/open-data-api) of the experimental MPDS data (5%): *CC BY 4.0*
- The closed part of the experimental MPDS data (95%): *commercial*


Citation
------

- Blokhin E, Villars P, Quantitative trends in physical properties of inorganic compounds via machine learning, [arXiv](https://arxiv.org/abs/1806.03553), **2018**
