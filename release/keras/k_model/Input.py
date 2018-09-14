from keras.utils import Sequence
from PIL import Image
import pickle
import os
import numpy as np
import uuid

from matplotlib import pyplot as plt

IMG_SIZE = 300
IMG_CHANEL = 3
class ImgInput(Sequence):
    """
    filePath：图片存储路径
    partOfFile：数据集包含的关键字，用于过滤数据集
    """
    def __init__(self, filePath=None,partOfFile="train"):
        self.filePath = filePath
        if filePath and os.path.isdir(filePath) and os.path.exists(filePath):
            self.files = [ i for i in os.listdir(filePath) if "label" not in i and partOfFile in i]
        else:
            self.files = []

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fileName = os.path.join(self.filePath,self.files[idx])
        # print("fileName",fileName)
        with open(fileName,"rb") as f:
            imgData = pickle.load(f)
        # print(imgData)
        # return (np.array(imgData["data"]),np.array(imgData["label"]))
        x = imgData["data"]
        y = imgData["label"]

        x = [np.reshape(item,[IMG_SIZE,IMG_SIZE,IMG_CHANEL]) / 255.0 for item in x ]
        x = np.array(x)
        x = x.astype(np.float32)
        y = np.array(y)

        return (x, y)

    def preprocess(self,img):
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img) / 255.0
        img = np.reshape(img, [1, IMG_SIZE, IMG_SIZE, IMG_CHANEL])
        return img

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

    def read_one(self):
        if os.path.isfile(self.filePath):
            img = Image.open(self.filePath)
            img = self.preprocess(img)
            print("img",img.shape)
            return img
        else:
            raise ValueError("please an absolute picture path")

if __name__ == "__main__":
    # filepath = []
    # imgInput = ImgInput("F:/tmp/tmp",partOfFile="train")
    # print(type(imgInput),imgInput.files,len(imgInput))
    # x = next(iter(imgInput))[0]
    # print(isinstance(x,list))
    # print(x[0].shape[0],x[0])
    # for i in imgInput:
    #     cnt = 0
    #     print(i[1])
    #     # print(i[0].shape)
    #     # print("*"*10)
    #     for j in i[0]:
    #         cnt += 1
    #         plt.imshow(j)
    #         plt.show()
    #         if cnt == 6:
    #             break
        # plt.imshow(i[0][0])
        # plt.show()

    imgInput = ImgInput(r"E:\PWORKSPACE\house3\data3")
    x, y = imgInput.getOneFile()
    for i in range(len(x)):
        print(y[i],type(x[i]),x[i].shape)
        plt.imshow(x[i])
        # plt.imsave(os.path.join(r"E:\PWORKSPACE\house3\data3\tmp",str(y[i])+"_"+str(uuid.uuid1())+".jpg"),x[i])
        plt.show()

    # filepath = r"E:\PWORKSPACE\house3\pic3\bedroom\0bbf71b6-a2d8-4703-9bb1-b5a9d699b70a.jpg"
    # imgInput = ImgInput(filepath)
    # print(imgInput.read_one())


