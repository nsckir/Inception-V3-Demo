import os
import os.path
import random
import json
from flask import Flask, request, g,  make_response
from werkzeug.utils import secure_filename
import inception_model

################################################
# Flask App configuration
################################################
# Simple, unsafe Flask settings
app = Flask(__name__, instance_relative_config=True)
app.config.from_object(__name__)
app.config.update({
    'DATABASE': os.path.join(app.root_path, 'serving.db'),
    'SECRET_KEY': 'development key',
    'USERNAME': 'admin',
    'PASSWORD': 'default',
    'UPLOAD_FOLDER': 'temp'
})

################################################
# Model initialization
################################################
if os.environ.get("FLASK_DEBUG") == "1":
    # This prevents multiple sessions from being created in DEBUG mode
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        sess = inception_model.Session()
else:
    sess = inception_model.Session()


################################################
# Functions for descriptions of model output
################################################
def create_descriptions():
    """Converts static/scoodit178_descriptions.txt into a list of strings"""
    descriptions = []
    with app.open_resource('static/scoodit178_descriptions.txt', mode='r') as f:
        for line in f:
            descriptions.append(line)
    return descriptions


def get_descriptions():
    """Returns the Inception-ResNet string description list"""
    if not hasattr(g, 'descriptions'):
        g.descriptions = create_descriptions()
    return g.descriptions


################################################
# Routes
################################################
@app.route('/', methods=['POST'])
def predict():
    """Classifies JPEG image passed in as POST data
    
    Assuming a JPEG file is passed in (as raw bytes), this function saves the 
    image to a the local temp directory, passes in the image to the TensorFlow
    model, and returns the top-5 guesses and path to the saved image to be 
    rendered to the client.

    NOTE: This function is NOT SAFE. Strictly for demonstration purposes. Does
    not do any safe-checking of the data being saved locally. Only use locally.
    """
    descr = []
    scores = []
    string_buffer = request.stream.read()
    feed_dict = {inception_model.get_input(sess): string_buffer}
    prediction = sess.run(inception_model.get_predictions(sess), feed_dict)
    top_k = prediction.argsort()[0][-5:][::-1]
    descriptions = get_descriptions()
    for idx in top_k:
        description = descriptions[idx]
        description = description.strip().split(':',)[1].rsplit('_', 1)[0].replace('_', ' ')
        score = prediction[0][idx]
        descr.append(description)
        scores.append(str(score))
    return make_response(json.dumps(zip(descr, scores)), 200)
