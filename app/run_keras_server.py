# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
import os

from keras.applications import ResNet50
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
import configparser
from flask import Flask, render_template, request, send_from_directory
from werkzeug import secure_filename

#import socket
#hostip = socket.gethostbyname(socket.gethostname())
#print("HOSTIP : ", hostip)


from keras.models import load_model

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

######################################################################
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

hostip = '192.168.0.21'
#hostip = '192.168.40.106'
config = configparser.ConfigParser()
config.read('/home/hieu/work/web/keras-rest/app/config/keras-rest.ini')
modelPath = config['SERVER']['modelPath']
print('path to model: ', modelPath)

app.config['UPLOADED_PHOTOS_DEST'] = config['SERVER']['imgSaveDir']



def predict2(filename) :
    data = {'success': False}

    if request.method == "POST" :
        image = np.array([img_to_array(load_img(filename))], 'f')
        data['predict'] = [a.tolist() for a in model.predict(image)]
        data['success'] = True
    return data
        
@app.route("/predict", methods = ["POST"])
def predict() :
    data = {'success': False}

    if request.method == "POST" :
        if request.files.get("image") :
            fPath = request.files.get("image")
            print('fPath: ', fPath)
            try :
                image = np.array([img_to_array(load_img(request.files.get("image")))], 'f')
                data['prediction'] = [a.tolist() for a in model.predict(image)]
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
        print (upload)
        filename = upload.filename

        ext = os.path.splitext(filename)[1]
        if (ext.strip().lower() in ['.jpg', '.png']) :
            print("processing")
        else :
            render_template("Error.html", message="File type not supported")

        destination = '/'.join([target, filename])
        imgFiles.append(filename)
        predictVals.append(predict2(destination)['predict'])
        upload.save(destination)
    img_pred = zip(imgFiles, predictVals)
    return render_template("gallery.html", img_pred=img_pred)
    #return render_template("gallery.html", image_names=imgFiles, predictVal=predictVals[0])

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
	app.run(host=hostip, debug=True)
