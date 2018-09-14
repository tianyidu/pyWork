import tensorflow as tf
import os
from multiprocessing import Process
from matplotlib import  pyplot as plt

class Pic:
    def __init__(self,old_folder,new_folder):
        self.ofd = old_folder
        self.nfd = new_folder
        self._HEIGHT = 224
        self._WEIGHT = 224
        # self.image = {}

    def init(self):
        if not os.path.exists(self.ofd):
            raise ValueError("dir not exists")
        if not os.path.exists(self.nfd):
            os.mkdir(self.nfd)
        os.chdir(self.ofd)
        for dir in os.listdir(self.ofd):
            if os.path.isdir(dir):
                if not os.path.exists(os.path.join(self.nfd,dir)):
                    os.mkdir(os.path.join(self.nfd,dir))

    def getImages(self,subfolder):
        images = []
        os.chdir(subfolder)
        for f in os.listdir(subfolder):
            if os.path.isfile(f):
                images.append(f)
        return images

    def resizeImage(self,image):
        if os.path.isfile(image) and image.endswith("jpg"):
            # print("resize:",image)
            image_data = tf.gfile.FastGFile(image,"rb").read()
            img = tf.image.decode_jpeg(image_data)
            img = tf.image.resize_images(img, [self._HEIGHT, self._WEIGHT])

            img = tf.cast(img,tf.uint8)

            return tf.image.encode_jpeg(img)

    def saveImage(self,file,image):
        with tf.gfile.FastGFile(file,"wb") as f:
            f.write(image)

    def run(self,path):
        with tf.Session() as sess:
            dir = os.path.join(self.ofd,path)
            for image in self.getImages(dir):
                print(os.getpid()," process ",path,image)
                oimgPath = os.path.join(self.ofd, path, image)
                resizedimg = self.resizeImage(oimgPath)
                nimgPath = os.path.join(self.nfd, path, image)
                print(oimgPath," save image to :",nimgPath)
                self.saveImage(nimgPath, resizedimg.eval())

    def start(self):
        self.init()
        for dir in os.listdir(self.ofd):
            # print("dir :",self.ofd,self.nfd,dir)
            # path = os.path.join(self.ofd,dir)
            if os.path.isdir(dir):
                t = Process(target=self.run,args=(dir,))
                t.start()
            else:
                self.run("")
            # print(t.name,path)
    # def start(self):
    #     self.init()
    #     with tf.Session() as sess:
    #         for dir in os.listdir(self.ofd):
    #             print("dir :",self.ofd,self.nfd,dir)
    #             for image in self.getImages(os.path.join(self.ofd,dir)):
    #                 oimgPath = os.path.join(self.ofd,dir,image)
    #                 resizedimg = self.resizeImage(oimgPath)
    #                 nimgPath = os.path.join(self.nfd,dir,image)
    #                 self.saveImage(nimgPath,resizedimg.eval())



if __name__ == "__main__":
    old_dir = "E:/PWORKSPACE/houseUtil/house/test"
    new_dir = "E:/PWORKSPACE/house3/pic224/test"
    pic = Pic(old_dir,new_dir)
    pic.start()

    # oldImg = r"E:\PWORKSPACE\houseUtil\test\midtmp"
    # newImg = r"E:\PWORKSPACE\houseUtil\test\change"
    # pic = Pic(oldImg,newImg)
    # pic.start()

