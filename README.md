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
- band gap (or its absence, _i.e._ whether a crystal is conductor or insulator)

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


Preparation for work
------

The model is trained on the MPDS data using the MPDS API and the scripts `train_regressor.py` and `train_classifier.py`. Some subset of the full MPDS data is opened and possible to obtain via MPDS API [for free](https://mpds.io/open-data-api). If the training is performed on the limited (_e.g._ opened) data subset, the scripts must be modified to make queries accordingly. The MPDS API returns an HTTP error code `402` if a user's request is authenticated, but not authorized. See a [full list of HTTP status codes](https://en.wikipedia.org/wiki/List_of_HTTP_status_codes).

The code tries to use the settings exemplified in a template:

```shell
cp data/settings.ini.sample data/settings.ini
```


Architecture and usage
------

The code can be used either as a *standalone command-line* application or as a *client-server* application.

Examples of the *standalone command-line* architecture are the scripts `mpds_ml_labs/test_props_cmd.py` and `mpds_ml_labs/test_design_cmd.py`.

In the case of the *client-server* architecture, the client and the server communicate over HTTP using a simple API, and any client able to execute HTTP requests is supported, be it a `curl` command-line client, a Python script or the rich web-browser user interface. Examples of the Python scripts are `mpds_ml_labs/test_props_client.py` and `mpds_ml_labs/test_design_client.py`.

Server part is a Flask app `mpds_ml_labs/app.py`. The simple HTML5 client apps `props.html` and `design.html`, supplied in the `webassets` folder, are served by a Flask app under `http://localhost:5000`. By default, to serve the requests the development Flask server is used. Therefore an _AS-IS_ deployment in an online environment without the suitable WSGI container is **highly discouraged**. For the production environments under the high load it is recommended to use something like [TensorFlow Serving](https://www.tensorflow.org/serving).


Used descriptor and model details
------

The term _descriptor_ stands for the compact information-rich representation, allowing the convenient mathematical treatment of the encoded complex data (_i.e._ crystalline structure). Any crystalline structure is populated to a certain relatively big fixed volume of minimum one cubic nanometer. Then the descriptor is constructed using the periodic numbers of atoms and the lengths of their radius-vectors. The details are in the file `mpds_ml_labs/prediction.py`.

As a machine-learning model an ensemble of decision trees ([random forest regressor](http://scikit-learn.org/stable/modules/ensemble.html)) is used, as implemented in [scikit-learn](http://scikit-learn.org) Python machine-learning toolkit. The whole MPDS dataset can be used for training. In order to estimate the prediction quality of the _regressor_ model, the _mean absolute error_ and _R2 coefficient of determination_ is saved. In order to estimate the prediction quality of the binary _classifier_ model, the _fraction incorrect_ (_i.e._ the _error percentage_) is saved. The evaluation process is repeated at least 30 times to achieve a statistical reliability.

Generating the crystal structure from the physical properties is done as follows. The decision-tree properties predictions of nearly 115k distinct MPDS phases are used for the radius-based neighbor learning. This allows to extrapolate the possible chemical elements for almost any given combination of physical properties. The results of the neighbor learning are approximately 7.6M rows, stored in a Postgres table `ml_knn`:

```sql
CREATE TABLE ml_knn (
    id  INT PRIMARY KEY,
    z   SMALLINT NOT NULL,
    y   SMALLINT NOT NULL,
    x   SMALLINT NOT NULL,
    k   SMALLINT NOT NULL,
    w   SMALLINT NOT NULL,
    m   SMALLINT NOT NULL,
    d   SMALLINT NOT NULL,
    t   SMALLINT NOT NULL,
    els VARCHAR(19)
);
CREATE SEQUENCE ml_knn_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER SEQUENCE ml_knn_id_seq OWNED BY ml_knn.id;
ALTER TABLE ONLY ml_knn ALTER COLUMN id SET DEFAULT nextval('ml_knn_id_seq'::regclass);
CREATE INDEX prop_z ON ml_knn USING btree(z);
CREATE INDEX prop_y ON ml_knn USING btree(y);
CREATE INDEX prop_x ON ml_knn USING btree(x);
CREATE INDEX prop_k ON ml_knn USING btree(k);
CREATE INDEX prop_w ON ml_knn USING btree(w);
CREATE INDEX prop_m ON ml_knn USING btree(m);
CREATE INDEX prop_d ON ml_knn USING btree(d);
CREATE INDEX prop_t ON ml_knn USING btree(t);
```

The full contents of this table can be provided by request. The found elements matching the given property ranges are used to compile a crystal structure based on the available MPDS structure prototypes (via the MPDS API). See `mpds_ml_labs/test_design_cmd.py`.


API
------

These are examples of using the `curl` command-line client.

For the local server:

```shell
curl -XPOST http://localhost:5000/predict -d "structure=data_in_CIF_or_POSCAR"
curl -XPOST http://localhost:5000/design -d "numerics=ranges_of_values_of_8_properties_in_JSON"
```

For the demonstration MPDS server:

```shell
curl -XPOST https://labs.mpds.io/predict -d "structure=data_in_CIF_or_POSCAR"
curl -XPOST https://labs.mpds.io/design -d "numerics=ranges_of_values_of_8_properties_in_JSON"
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
