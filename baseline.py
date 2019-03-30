#baseline.py modified
import numpy
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
#import massageData

CLASS_NUM = 3
BITMAP_DIM=784
BATCH_SIZE=100
EPOCHS=2
model=None
def baseline_model():
  # create model
  # model.add(Dense(1, input_dim=BITMAP_DIM, activation='sigmoid'))
  # model.add(Dense(CLASS_NUM, activation='softmax'))
  # # Compile model
  # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  model = Sequential()
  model.add(Dense(units=600, activation='relu', input_dim=784))
  model.add(Dropout(0.3))
  model.add(Dense(units=400, activation='relu'))
  model.add(Dropout(0.3))
  model.add(Dense(units=100, activation='relu'))
  model.add(Dropout(0.3))
  model.add(Dense(units=25, activation='relu'))
  model.add(Dropout(0.3))
  model.add(Dense(units=CLASS_NUM, activation='softmax'))
  model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

def main():
  print ("Done loading the libraries")
  t0 = time.time()
  seed = 7
  numpy.random.seed(seed)
  data = messageData()
  data.setLengths()
  #generators
  train_generator = data.generator('data/numpy_bitmap/', isTrain=True, batch_size=BATCH_SIZE) 
  validation_generator = data.generator('data/numpy_bitmap/', isTrain=False, batch_size=BATCH_SIZE)
  print("length of train_samples: ",data.total_len_train)
  print("length of validation_samples: ",data.total_len_valid)
  model = baseline_model()
  model.fit_generator(train_generator, 
                  samples_per_epoch=data.total_len_train,
                  validation_data=validation_generator,
                  nb_val_samples=data.total_len_valid, epochs=EPOCHS, steps_per_epoch=BATCH_SIZE, verbose=1)
  print("Model training, testing completed- Saving Model...")
  model.save("doodle_trial_model.h5")
if __name__ == '__main__':
  main()