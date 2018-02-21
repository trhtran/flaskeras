# USAGE
# python simple_request.py

# import the necessary packages
import sys
import requests

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://192.168.40.106:5000/predict"
IMAGE_PATH = sys.argv[1]

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}
# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# ensure the request was sucessful
if r["success"]:
    print ("PREDICTION: ", r['prediction'])
# otherwise, the request failed
else:
	print("Request failed")
