Data-driven predictions from the crystalline structure
======

Rationale
------

This is the proof of concept, how a relatively unsophisticated statistical model (namely, random forest regressor) trained on the large MPDS dataset predicts a set of physical properties from the only crystalline structure. Similarly to _ab initio_, it could be called _ab datum_. (Note however that the simulation of physical properties with a comparable precision normally takes days, weeks or even months, whereas the present prediction method takes less than a second.) A crystal structure in either CIF or POSCAR format is required.

Architecture
------

This is a client-server application. The client is not required although, and it is possible to employ the server code as a standalone command-line application. The client is used for a more convenient demonstration only. The client and the server communicate using HTTP. Any client able to execute HTTP requests is supported, be it a `curl` command-line client or rich web-browser user interface. As an example of the latter, a simple HTML5 app `index.html` is supplied. Server part is a Flask app, loading the pre-trained ML models:

```python
python index.py /tmp/path_to_model_one /tmp/path_to_model_two
```

Web-browser user interface is then available under `http://localhost:5000`. To serve the requests the development Flask server is used. Therefore an _AS-IS_ deployment in an online environment without the suitable WSGI container is highly discouraged. Serving of the ML models is very simple. For the production environments under high load it is recommended to follow e.g. [TensorFlow Serving](https://www.tensorflow.org/serving).

API
------

```shell
curl -XPOST http://localhost:5000/predict -d "structure=data_in_CIF"
curl -XPOST https://tilde.pro/labs/predict -d "structure=data_in_POSCAR"
```
