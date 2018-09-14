#coding:utf8
import tensorflow as tf
from matplotlib import pyplot as plt
"""
读取图片类
"""

class Img:
    def __init__(self,image):
        self.image = image
        self._SIZE = 300
        self._CHANEL = 3

    #预处理图片，将图片转换为统一大小，输出可预测的数据格式
    def preProcess(self):
        fr = tf.gfile.FastGFile(self.image, "rb").read()
        img = tf.image.decode_jpeg(fr)
        img = tf.image.resize_images(img, [self._SIZE, self._SIZE])
        img = tf.cast(img, tf.uint8)
        img = tf.reshape(img, [-1, self._SIZE, self._SIZE, self._CHANEL])
        label = tf.constant(1,shape=[1,1])
        return img,label

    #预测完成后处理方法
    def afterProcess(self):
        pass

# img = Img(r"E:\PWORKSPACE\house3\test.jpg")
# img.preProcess()