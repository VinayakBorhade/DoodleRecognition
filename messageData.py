#messageData.py modified
import numpy as np
import os
import random

class messageData:
  def __init(self):
    self.total_len_train=0
    self.total_len_valid=0
  
  def setLengths(self, folder='data/numpy_bitmap/'):
    self.total_len_train=0
    self.total_len_valid=0
    self.folder=folder
    flag=True
    for i, filename in enumerate(os.listdir(self.folder)):
      fullpath = self.folder + filename
      name = filename.split('.')[0]
      samples = np.load(fullpath, mmap_mode='r')
      s_temp=int(samples.shape[0]*0.1)
      self.total_len_train+= s_temp*0.7
      self.total_len_valid+= s_temp*0.3
      if flag:
        flag=False
        print("dimension of the image: ",samples[0].shape)
        
    self.total_len_train=int(self.total_len_train)
    self.total_len_valid=int(self.total_len_valid)
  
  def generator(self, folder='data/numpy_bitmap/', isTrain=True, batch_size=32):
    self.folder=folder
    """
    Yields the next training batch.
    Suppose `samples` is an array [[image1_filename,label1], [image2_filename,label2],...].
    samples is an array of 28*28 image [image1, image2,....]
    """
    while True:
      for i, filename in enumerate(os.listdir(self.folder)):
        fullpath = self.folder + filename
        name = filename.split('.')[0]
        samples_full = np.load(fullpath, mmap_mode='r')
        samples_full_len = samples_full.shape[0]
        samples = samples_full[0:int(0.1*samples_full_len)]
        samples_len = samples.shape[0]
        samples_local=np.array([])
        start,end=0,0
        if isTrain:
          samples_local=samples[0:int(0.7*samples_len)]
          start,end=0,int(0.7*samples_len)
        else:
          samples_local=samples[int(0.7*samples_len):samples_len]
          start,end=0,int(0.3*samples_len)

        X,Y=np.array([]),[]
        np.random.permutation(samples_local)
        for offset in range(start, end, batch_size):
          X=samples_local[offset:offset+batch_size]
          Y=[i]*int(X.shape[0])
          yield X, Y

data=messageData()
data.setLengths()
print("total_len_train", data.total_len_train)
print("total_len_valid", data.total_len_valid)