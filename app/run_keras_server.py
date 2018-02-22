# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
import os
import io

from keras.applications import ResNet50
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import configparser
import flask
from flask import Flask, render_template, request, send_from_directory
from flask_login import LoginManager
from flask_login import login_required
from werkzeug import secure_filename

import utils

#import socket
#hostip = socket.gethostbyname(socket.gethostname())
#print("HOSTIP : ", hostip)


from keras.models import load_model

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
#app = flask.Flask(config)
login_manager = LoginManager(app)
model = None

######################################################################
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

#TODO use config.py instead of configparser
#hostip = '192.168.0.21'
hostip = '192.168.40.106'
config = configparser.ConfigParser()
config.read('%s/%s' % (APP_ROOT, 'config/keras-rest.ini') )
modelPath = '%s/%s' % (APP_ROOT, config['SERVER']['modelPath'])
print('path to model: ', modelPath)

app.config['UPLOADED_PHOTOS_DEST'] = config['SERVER']['imgSaveDir']

######################################################################
def getPrediction(filename) :
    ''' call model for prediction, transform arrays '''
    image = np.array([img_to_array(load_img(filename))], 'f')
    angle_binned, throttle = model.predict(image)
    angle_unbinned = utils.linear_unbin(angle_binned)
    return [angle_unbinned, float(throttle[0][0])]


def predict2(filename) :
    ''' function to get prediction inside the program, w/o jsonification '''
    data = {'success': False}
    if request.method == "POST" :
        try :
            data['predict'] = getPrediction(filename)
            data['success'] = True
        except Exception as reason :
            print("exception: %s" % (reason))
    return data

@app.route("/predict", methods = ["POST"])
def predict() :
    data = {'success': False}

    if request.method == "POST" :
        if request.files.get("image") :
            try :
                data['predict'] = getPrediction(request.files.get("image"))
                data['success'] = True
            except Exception as reason :
                print("exception: %s" % (reason))
    return flask.jsonify(data)

@app.route('/')
def index() :
    return render_template("upload.html")


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    target = os.path.join(APP_ROOT, config['SERVER']['imgSaveDir'])
    if not os.path.isdir(target) :
        os.mkdir(target)

    imgFiles = []
    predictVals = []
    for upload in request.files.getlist("file") :
        app.logger.info(upload)
        filename = upload.filename

        ext = os.path.splitext(filename)[1]
        if (ext.strip().lower() in ['.jpg', '.png']) :
            app.logger.debug("processing")
        else :
            render_template("Error.html", message="File type not supported")

        destination = '/'.join([target, filename])
        upload.save(destination)
        imgFiles.append(filename)
        predictVals.append(predict2(destination)['predict'])
    img_pred = zip(imgFiles, predictVals)
    return render_template("gallery.html", img_pred=img_pred)

@app.route('/<filename>')
def send_image(filename):
    return send_from_directory(config['SERVER']['imgSaveDir'], filename)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    model = load_model(modelPath)
    model._make_predict_function()
    app.secret_key = os.urandom(12)
    app.run(host=hostip, debug=True)
