from keras.utils import Sequence
import pickle
import os
import numpy as np
import random

IMG_SIZE = 300
IMG_CHANEL = 3
class ImgInput(Sequence):
    """
    filePath：图片存储路径
    partOfFile：数据集包含的关键字，用于过滤数据集
    """
    def __init__(self, filePath,partOfFile="train"):
        self.filePath = filePath
        if os.path.exists(filePath):
            self.files = [ i for i in os.listdir(filePath) if "label" not in i and partOfFile in i]
        else:
            self.files = []
        random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        #print("filename ",self.files[idx])
        fileName = os.path.join(self.filePath,self.files[idx])
        with open(fileName,"rb") as f:
            imgData = pickle.load(f)
        # print(imgData)
        # return (np.array(imgData["data"]),np.array(imgData["label"]))
        x = imgData["data"]
        y = imgData["label"]

        x = [np.reshape(item,[IMG_SIZE,IMG_SIZE,IMG_CHANEL]) / 255.0 for item in x ]
        x = np.array(x).astype(np.float32)
        y = np.array(y).astype(np.float32)
        #print("xy",x[0],y)    
        return (x,y)

    def getOneFile(self):
        fileName = os.path.join(self.filePath,self.files[0])
        x = None
        y = None
        if os.path.exists(fileName):
            with open(fileName,"rb") as f:
                imgdata = pickle.load(f)

            data = imgdata["data"]
            label = imgdata["label"]

            x= np.array(data).astype(np.float32)
            x = np.reshape(x,[len(data), IMG_SIZE, IMG_SIZE, IMG_CHANEL]) / 255.0
            y = np.array(label)

        return (x,y)

if __name__ == "__main__":
    imgInput = ImgInput(r"/home/app_user_5i5j/workspace/pwork/lj_k/data")
    print(imgInput.filePath,imgInput.files,os.path.exists(imgInput.filePath))
