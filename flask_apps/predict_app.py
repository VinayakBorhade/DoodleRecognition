import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import request, jsonify, Flask
app=Flask(__name__)
def get_model():
    global model
    model=load_model("doodle_trial_model.h5")
    print(" * Model loaded!")
def preprocess_image(image, target_size):
    image=image.resize(target_size)
    image=img_to_array(image)
    image=np.expand_dims(image, axis=0)
    return image
print(" * Loading Keras model...")
get_model()
@app.route("/predict", methods=["POST"])
def predict():
    message=request.get_json(force=True)
    encoded=message['image']
    decoded=base64.b64decode(encoded)
    image=Image.open(io.BytesIO(decoded))
    processed_image=preprocess_image(image, target_size=(784))
    prediction=model.predict(processed_image).tolist()
    response={
        'prediction':{
            'airplane':prediction[0][0],
            'apple':prediction[0][1],
            'banana':prediction[0][2]
        }
    }
    return jsonify(response)
if __name__=="__main__":
	app.run()
