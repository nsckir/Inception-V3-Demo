# Pretrained Inception-V3 Demo with Flask

## Quick note

This demo is _not_ supposed to represent a production-ready server.
It is not secure for the web, and is not setup to be responsive (server-client responses 
and model evaluation run in the same thread instead of asyncronously). Rather, this is a quick 
demonstration of how to utilize a pretrained Inception-V3 model and quickly put together a
 prototype with it.

## Dependencies

* [NumPy](http://www.numpy.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [Flask](http://flask.pocoo.org/)
* [werkzeug](http://werkzeug.pocoo.org/)

## Run it!

1. Make sure you have the above dependencies installed (`pip install ...`)
2. Clone the repo, move into the directory, and execute `run_server.sh`:

```
$ git clone https://nsckir@bitbucket.org/nsckir/scoodit_server.git
$ cd inception-flask-demo
$ ./run_server.sh
```

## About the files

### `run_inception_server.sh`

* The main file, runs the preprocessing of files and then launches the Flask server
* You must run this script from its directory, not from a parent or child directory. ie. this:
```
$ ./run_inception_server.sh
```
not:
```
$ ../run_inception_server.sh
```
or:
```
$ inception-flask-demo/run_inception_server.sh
```

### Pre-processing files

#### `inception_v3_export.py`

* Converts the trained model into a frozen and optimized protobuf file in `serving/static`

### Flask Server files

#### `serving/serving_inception.py`

* Starts up the Flask server and TensorFlow model Session
* Contains route functions for the server

#### `serving/inception_model.py`

* Functions and singleton Session class for the Inception-ResNet Model
* Singleton is used to preserve optimizations and prevent users from reloading the model from memory

--- 

### TensorFlow files.

The [TENSORFLOW_LICENSE](TENSORFLOW_LICENSE) applies to the following files:

#### `inception_v3_tf1.py`

* Constructs the Inception V3 model
* From the [TensorFlow Models repository](https://github
.com/tensorflow/models/blob/master/slim/nets/inception_v3.py)

#### `inception_preprocessing.py`

* Creates image steps to preprocess image data for both training and
inference
* [From the TensorFlow Models repository](https://github.com/tensorflow/models/blob/master/slim/preprocessing/inception_preprocessing.py)

#### `freeze_graph.py`

* Tool for "freezing" a TensorFlow graph. Converts Variable objects into constants.
* [From the TensorFlow repository](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py)

#### `optimize_for_inference.py`

* Tool that removes unnecessary operations from a graph.
* [From the TensorFlow repository](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference.py)

#### `optimize_for_inference_lib.py`

* Utility functions needed for `optimize_for_inference.py`
* [From the TensorFlow repository](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference_lib.py)
