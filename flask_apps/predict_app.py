import base64
import numpy as np
import io
from PIL import Image
import imageio
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
	global graph
	graph = tf.get_default_graph()

#def preprocess_image(image, target_size):
#	image=image.resize(target_size)
#	image=img_to_array(image)
#	image=np.expand_dims(image, axis=0)
#	return image
print(" * Loading Keras model...")
get_model()
@app.route("/predict", methods=["POST"])
def predict():
	message=request.get_json(force=True)
	encoded=message['image']
	decoded=base64.b64decode(encoded)
	image=Image.open(io.BytesIO(decoded))
	#processed_image=preprocess_image(image, target_size=(28,28))
	image.resize((80, 80), Image.ANTIALIAS)
	image=np.array(image)
	#image_np=numpy.array(image)
	#print("before image slicing!!!!!!")
	#print(image)
	image=image[:,:,0]
	#print("after image slicing!!!!!!")

	image=image.flatten()
	image_np=np.array([image])
	#print("image_np.shape")
	#print(image_np.shape)
	#print(image_np)
	#model=load_model("doodle_trial_model.h5")
	with graph.as_default():
		prediction=model.predict(image_np).tolist()
		print("after prediction")
		print(prediction)
		response={
			'prediction':{
				'apple':prediction[0][0],
				'banana':prediction[0][1],
				'circle':prediction[0][2],
				'pineapple':prediction[0][3]
			}
		}
		#print("made prediction response: ", prediction)
	return jsonify(response)
if __name__ == '__main__':
	app.run(debug=False)
