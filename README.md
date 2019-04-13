# Doodle Recognition
Watch a simple neural network  recognize / classify hand-drawn Fruit drawings

Dataset to use is the Google Quick Draw dataset
Kaggle Link - https://www.kaggle.com/c/quickdraw-doodle-recognition/
Github Link - https://github.com/googlecreativelab/quickdraw-dataset 

<img src="https://github.com/VinayakBorhade/DoodleRecognition/blob/master/samples/results.PNG" width="50%" />

## Requirements
* Python
* Tensorflow
* Keras
* numpy
* scikit-learn
* PIL
* scipy
* Flask
* JQuery

Download and move into the project directory
```
$ git clone https://github.com/VinayakBorhade/DoodleRecognition.git
$ cd DoodleRecognition
```

Run the app with flask server
```
$ chmod u+x run_server.sh
$ ./run_server.sh
```

Once server is started, open the UI for app in your browser by entering the follo. link- 
```
localhost:5000/static/predict.html
```

## Dowonloading Dataset

The Google quickdraw dataset is hosted on Google Cloud Storage

To get a copy of it, you need to use gsutil to download
* Install gsutil from here: https://cloud.google.com/storage/docs/gsutil_install#install
```
$ curl https://sdk.cloud.google.com | bash
$ exec -l $SHELL
$ gcloud init
```

* Download all the image drawings from the dataset
```
$ ./download_fulldataset.sh
```

The fulldataset is big (~37GB). For initial testing, we would be using a small subset

```
$ ./download_minidataset.sh
```

## Building the Model