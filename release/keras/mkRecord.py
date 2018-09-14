from PIL import Image
import pickle
import numpy as np
import os
import random
from matplotlib import pyplot as plt

class PicUtil:
    """
    filePaht : 图片存储顶层目录，该目录下应包含各个分类文件夹
    fileName : 图片数据集保存文件名前缀
    batch_size : 每个数据集大小，也为后期训练使用数据集大小
    size : 每个分类大小，如不指定，则为每个分类的最小数量
    """
    def __init__(self,filePath,fileName,batch_size=None,size=None):
        self.filePath = filePath
        self.fileName = fileName
        self.batch_size = batch_size
        self.size = size

    #将每个文件夹下图片名称和类别保存为字典
    def getImgDict(self):
        catagory = []
        catagory_items = {}
        for dir in os.listdir(self.filePath):
            absolutePath = os.path.join(self.filePath,dir)
            #去除test目录，test目录为测试集所在目录
            if os.path.isdir(absolutePath) and dir != "test":
                catagory.append(absolutePath)
                #保存每个类别和对应的图片到字典
                catagory_items[absolutePath] = os.listdir(absolutePath)
                if self.size == None:
                    #每个类别下最小的图片数量
                    self.size = min(len(catagory_items[absolutePath]),0)
        # print("size ",self.size,catagory_items)
        return catagory,catagory_items

    # 将图片转为一维数组
    def getImgArray(self, fileName):
        img = Image.open(fileName)
        reshape = np.reshape(img, [-1])
        return reshape

    # 保存数据集
    # imgdata : 图片字典，数据结构为：{图片目录:[图片名称]}
    # batch_idx : 每个批次的序号，生成文件名后缀
    # catagory : 中文类别列表
    def save(self, batch_idx, imgBatch,catagory):
        with open(self.fileName + "_" + str(batch_idx), "wb") as f:
            pickle.dump(imgBatch, f, 2)

        labelFile = self.fileName + "_label"
        if not os.path.exists(labelFile):
            with open(labelFile,"w") as f:
                f.write(str(catagory))

    #制作批数据集
    def getBatch(self):
        catagory, catagory_items = self.getImgDict()
        #保存每个类别下以使用的index索引值
        index = {}
        for key in catagory_items.keys():
            index[key] = 0
        #循环处理每一批次数据
        for batch_idx in range(int(np.floor(self.size/self.batch_size))):
            data = {"data":[],"label":[]}
            for id in range(self.batch_size):
                catagory_id = id % len(index.keys())
                catagory_key = catagory[catagory_id]
                try:
                    # if data.get(catagory_key) == None:
                    #     data[catagory_key] = [ catagory_items[catagory_key][index[catagory_key]] ]
                    # else:
                    #     data[catagory_key].append(catagory_key,catagory_items[catagory_key][index[catagory_key]])
                    fileName = os.path.join(catagory_key,catagory_items[catagory_key][index[catagory_key]])
                    data["data"].append(self.getImgArray(fileName))
                    data["label"].append(catagory_id)
                    index[catagory_key] = index[catagory_key] + 1
                except Exception as e:
                    print("error",catagory_key,catagory,index[catagory_key],e)
            print(data["label"])
            print("save batch ",batch_idx)
            self.save(batch_idx,data,catagory)

    #产生多个文件的数据集
    def run(self):
        self.getBatch()

    #将文件归档到一个数据集
    def mkOneRc(self,each_size=600):
        dirs = os.listdir(self.filePath)
        if "test" in dirs:
            dirs.remove("test")
        imgs = []
        label = []
        for dir in dirs:
            label.append(dir)
            absolutePath = os.path.join(self.filePath,dir)
            loop = 0
            for img in os.listdir(absolutePath):
                if loop < 600:
                    data = {}
                    imgfile = os.path.join(absolutePath,img)
                    data[imgfile] = dir
                    imgs.append(data)
                    loop += 1
        random.shuffle(imgs)
        print(label,imgs)
        imgdatas = {"data":[],"label":[]}
        for img in imgs:
            for k,v in img.items():
                imgarray = self.getImgArray(k)
                imglabel = label.index(v)
                imgdatas["data"].append(imgarray)
                imgdatas["label"].append(imglabel)

        self.save(1800,imgdatas,label)

if __name__=="__main__":

    dataDir = r"E:\PWORKSPACE\house3\pic224"
    testDir = r"E:\PWORKSPACE\house3\pic224\test"
    train_destFile = r"E:\PWORKSPACE\house3\data224\train_data_batch"
    eval_destFile = r"E:\PWORKSPACE\house3\data224\eval_data_batch"
    #
    # pic = PicUtil(dataDir,train_destFile,1800,1800)
    # pic.run()
    #
    # pic = PicUtil(testDir,eval_destFile,150,150)
    # pic.run()

    # with open(r"E:\PWORKSPACE\house3\data224\train_data_batch_32","rb") as f:
    #     img = pickle.load(f)
    # for i in range(20):
    #     image = img["data"][i]
    #     print(img["label"][i])
    #     # print(img,len(img))
    #     image = np.reshape(image,[224,224,3])
    #     plt.imshow(image)
    #     plt.show()

    pic = PicUtil(dataDir,train_destFile)
    pic.mkOneRc()

    pic = PicUtil(testDir,eval_destFile)
    pic.mkOneRc()

    # img = Image.open(r"E:\PWORKSPACE\houseUtil\resized_2\bedroom\0bbf71b6-a2d8-4703-9bb1-b5a9d699b70a.jpg")
    # img = np.array(img)
    # print(img,img.shape)