import numpy as np
import os
#import myConstants as mc

from sklearn.model_selection import train_test_split

class messageData:
  def getData(self):
    """ Returns the full dataset. This is 100% of the loaded data. """
    return (self.X, self.Y)

  def getTrain(self):
    """ Returns a tuple x,y for the training dataset. """
    return (self.X_train, self.y_train)

  def getTest(self):
    """ Returns a tuple x,y for the test dataset. """
    return (self.X_test, self.y_test)

  def getDev(self):
    """ Returns a tuple of x,y for the developmetn dataset. """
    return (self.X_dev, self.y_dev)

  def __init__(self, folder='data/numpy_bitmap/'):
    self.BITMAP_DIM=784
    self.folder = folder
    self.X = np.zeros((0, self.BITMAP_DIM))
    self.Y = []
    self.X_train = np.zeros((0, self.BITMAP_DIM))
    self.y_train = []
    self.X_dev = np.zeros((0, self.BITMAP_DIM))
    self.y_dev = []
    self.X_test= np.zeros((0, self.BITMAP_DIM))
    self.y_test = []
    random_seed = 42
    for filename in os.listdir(self.folder):
      fullpath = self.folder + filename
      # use filename as Y label.
      name = filename[:-4]
      x = np.load(fullpath)
      y = [name] * x.shape[0]
      # Full dataset.
      self.X = np.concatenate((self.X, x), axis=0)
      self.Y += y
      # Split into train/test/dev. See also comment above.
      X_train, X_eval, y_train, y_eval = train_test_split(x, y, test_size=0.2,random_state=random_seed)
      X_dev, X_test, y_dev, y_test = train_test_split(X_eval, y_eval, test_size=0.5, random_state=random_seed)
      # Train split.
      self.X_train = np.concatenate((self.X_train, X_train), axis=0)
      self.y_train += y_train
      # Dev split.
      self.X_dev = np.concatenate((self.X_dev, X_dev), axis=0)
      self.y_dev += y_dev
      # Test split.
      self.X_test = np.concatenate((self.X_test, X_test), axis=0)
      self.y_test += y_test

    # Convert into NP Arrays.
    self.Y = np.array(self.Y)
    self.y_train = np.array(self.y_train)
    self.y_test = np.array(self.y_test)
    self.y_dev = np.array(self.y_dev)

    print("All: ", self.X.shape)
    print("train: ", self.X_train.shape)
    print("Test: ", self.X_test.shape)
    print("Dev: ", self.X_dev.shape)


def main():
	data = messageData()

if __name__ == '__main__':
	main()
