import numpy as np
import os
'''
class messageData:
    BITMAP_DIM=784
    def __init__(self, folder='data/numpy_bitmap/'):
        self.folder = folder
        self.X=np.empty((3,784))
        self.Y=[]
        for i,filename in enumerate(os.listdir(self.folder)):
            fullpath=self.folder+filename
            #use filename as Y
            name=filename[:-4]
            x=np.load(fullpath,mmap_mode='r')
            print("x.shape[0]: ",x.shape[0])
			size=x.shape[0]
            y=[name for i in range(size)]
			reset=False
			end=0
			while not reset:
				start=end
				end+=batch_size
				if end>size:
					end=size
					reset=True
				yield x[start:end],y[start:end]
            self.X[i]=x
            #print(self.X)
            self.Y =self.Y + y
        self.Y=np.array(self.Y)
    def getX(self):
        return self.X
    def getY(self):
        return self.Y
'''
def generate(batch_size):
	folder='data/numpy_bitmap/'
	for i, filename in enumerate(os.listdir(folder)):
		fullpath = folder + filename
		name = filename.split('.')[0]
		x = np.load(fullpath, mmap_mode='r')
		size = x.shape[0]
		print("inside generate, name: ",name," , size",size)
		y = [name for i in range(size)]
		reset = False
		end = 0
		while not reset:
			start = end
			end += batch_size
			if end  > size:
				end = size
				reset = True
			yield x[start: end], y[start: end]

def main():
	#USE
	total_size=0
	X,Y=[],[]
	f=True
	for x, y in generate(batch_size=64):
		'''
		print("x: ",x)
		print("y: ",y)
		print("x.size(): ",len(x))
		print("y.size(): ",len(y))
		'''
		
		total_size+=len(x)
		if f:
			f=False
			print("type(X): ", type(X))
			print("type(x): ", type(x))
		
		#X+=x.astype(int)
		X.append(x)
		#Y=Y+y
	print("total_size: ",total_size)
	print("X.size: ",len(X))
	print("Y.size: ",len(Y))
		#Forward propagate
		#where x, y are minibatches of size (64, 784)
    
	#data=messageData()
    #print("x values: ",data.getX().shape)
    #print("y values: ",data.getY().shape)
	
if __name__=='__main__':
    main()