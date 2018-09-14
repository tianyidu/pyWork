import tensorflow as tf
import os
from PIL import Image
from matplotlib import pyplot as plt
import random
from datetime import datetime
import time

dataDir = r"E:\PWORKSPACE\houseUtil\resized_1"
testDir = r"E:\PWORKSPACE\houseUtil\resized_1\test"
train_destFile = r"E:/PWORKSPACE/houseUtil/resized_1/train.tfc"
eval_destFile = r"E:\PWORKSPACE\houseUtil\resized_1\eval.tfc"

classes = {"kitchen","wc","bedroom","livingroom"}

def createRecord(dirPath,destFile):
    label = []
    images = []
    writer = tf.python_io.TFRecordWriter(destFile)
    for index,name in enumerate(classes):
        print(index,name)
        """
        1 wc
        2 bedroom
        0 kitchen
       """
        # label.append(index)
        path = os.path.join(dirPath,name)
        os.chdir(path)
        img_num = 0
        for file in os.listdir(path):
            # print("file:",file)
            image = Image.open(file)
            image = image.tobytes()
            # example = tf.train.Example(features = tf.train.Features(feature={
            #                                         # "name":tf.train.Feature(bytes_list = tf.Train.BytesList(value=[file])),
            #                                         "label":tf.train.Feature(int64_list = tf.train.Int64List(value=[index])),
            #                                         "image_raw":tf.train.Feature(bytes_list = tf.train.BytesList(value=[image]))
            #                                     }))
            # writer.write(example.SerializeToString())
            img_num += 1
            if img_num <= 600:
                imageDic = {"label": index, "image_raw": image}
                images.append(imageDic)

    print("images:",len(images))
    for j in range(int(len(images)/len(classes))):
        for i in range(len(classes)):
            index = j + int(len(images)/len(classes)) * i
            # if index == len(images):
            #     index = index - 1
            # if len(images) > 0:
            # print(j,index)
            img = images[index]
            example = tf.train.Example(features = tf.train.Features(feature={
                                                    # "name":tf.train.Feature(bytes_list = tf.Train.BytesList(value=[file])),
                                                    "label":tf.train.Feature(int64_list = tf.train.Int64List(value=[img["label"]])),
                                                    "image_raw":tf.train.Feature(bytes_list = tf.train.BytesList(value=[img["image_raw"]]))
                                                }))
            writer.write(example.SerializeToString())

    writer.close()

def readRecord(filename):
    reader = tf.TFRecordReader()
    print("filename: ",filename)
    file_queue = tf.train.string_input_producer([filename])
    index,serialized_example = reader.read(file_queue)
    print("index: ",index,serialized_example)
    features = tf.parse_single_example(serialized_example,
                                       features = {
                                                        "label":tf.FixedLenFeature([],tf.int64),
                                                        "image_raw":tf.FixedLenFeature([],tf.string)
                                                    })
    image = tf.decode_raw(features["image_raw"],tf.uint8)
    image = tf.reshape(image,[300,300,3])

    print("image: ",image)
    # image = tf.cast(image,tf.float32)
    # image = tf.image.per_image_standardization(image)
    image = (image - 128) / 255

    print("image1:",image)
    label = tf.cast(features["label"],tf.int32)
    return image,label

def createBatch(filename,batchsize):
    image,label = readRecord(filename)
    min_after_deque = 1
    capacity = min_after_deque + 1 * batchsize
    image_batch,label_batch = tf.train.shuffle_batch([image,label],
                                                     batch_size=batchsize,
                                                     capacity=capacity,
                                                     min_after_dequeue=min_after_deque)
    # label_batch = tf.one_hot(label_batch,depth=4)
    print("label ",label_batch)
    return image_batch,label_batch

def create():
    createRecord(dataDir, train_destFile)
    createRecord(testDir, eval_destFile)

def read():
    with tf.Session() as sess:
        image_batch, label_batch = createBatch(filename=train_destFile, batchsize=8)
        # sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))
        sess.run(tf.local_variables_initializer())
        tf.train.start_queue_runners(sess=sess)
        label,image = sess.run([label_batch,image_batch])
        for i in range(len(label)):
            try:
                plt.imshow(image[i])
                plt.show()
                # for i in range(len(image)):
                #     plt.imsave(fname=os.path.join("E:/PWORKSPACE/houseUtil/test/midtmp",str(int(datetime.now().timestamp()))+str(i)+"_"+str(label[i])+".jpg"),arr=image[i])
                #     time.sleep(1)

                print("label_batch: ",label)
                # print("label_batch: ",label," image_batch:",image[0])
            except Exception as e:
                print("error",e)

if __name__ == "__main__":
    # create()

    # read()

    img = Image.open(r"E:\PWORKSPACE\house3\pic\wc\07a2564e-e483-4387-9c91-35df728d86fe.jpg").convert("L")
    img.show()


