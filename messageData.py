import numpy as np
import os
class messageData:
    BITMAP_DIM=784
    def __init__(self, folder='data/numpy_bitmap/'):
        self.folder = folder
        self.X=np.zeros((0,784))
        self.Y=[]
        for filename in os.listdir(self.folder):
            fullpath=self.folder+filename
            #use filename as Y
            name=filename[:-4]
            x=np.load(fullpath)
            y=[name]*x.shape[0]
            #print(self.X)
            self.X=np.concatenate((self.X,x),axis=0)
            #print(self.X)
            self.Y =self.Y + y
        self.Y=np.array(self.Y)
    def getX(self):
        return self.X
    def getY(self):
        return self.Y

def main():
    data=messageData()
    print("x values: ",data.getX().shape)
    print("y values: ",data.getY().shape)

if __name__=='__main__':
    main()