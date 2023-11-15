from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
# import tensorflow as tf
import os
import numpy as np
import base64
import cv2
from util import convertToJpg, allowed_file

# import model functionality

'''try to keep model functionality separate from server.py,
    if possible, creaete a modal.py file and import 
    a single function doing all the work <3,
'''


UPLOAD_FOLDER = os.path.basename('uploads')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key = "6575fae36288be6d1bad40b99808e37f"
# can be any string(not empty)


classes = ["normal", "cancer"]  # place it in modal.py if seems appropriate


@app.route("/", methods=["POST"])
def process_image():
    image_recieved = request.get_json()

    filename = secure_filename(image_recieved['filename'])
    imgdata = base64.b64decode(image_recieved['data'])

    if (not imgdata) and (filename == '') and (not allowed_file(filename)):
        return jsonify({'message': "Invalid Image.", 'code': 400})
        # code: 400 for bad request

    with open(os.path.join(app.config['UPLOAD_FOLDER'],
                           filename), 'wb') as f:
        f.write(imgdata)

    # use this as image path for model
    filepath_for_model = convertToJpg(app, filename)

    response = {}

    '''
    model_response = Call_Model(filepath_for_model) #call model functionality
    response['message'] = model_response  #add model response to response obj
    response['further_details'] = ... # add further details about result as needed
    response['code'] = 200 #succesfull response
    '''
    response['message'] = "Model Prediction."  # for sake of testing
    response['code'] = 200

    return jsonify(response)


@app.route("/", methods=["GET"])
def render_index():
    return render_template("index.html")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port='5000')
