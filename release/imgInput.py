import tensorflow as tf
import cv2

IMAGE_SIZE = 300
IMAGE_CHANEL = 3

def one_read(filename):
    img = cv2.imread(filename)
    # cv2.imshow("img",img)
    # cv2.waitKey(0)
    # print("img1: ",img.shape)
    img = tf.cast(img,tf.float32)
    # img = tf.image.per_image_standardization(img)
    img = tf.reshape(img,[-1,IMAGE_SIZE,IMAGE_SIZE,IMAGE_CHANEL])
    img = img / 255
    # print("img2: ",img)
    return img
#read image from file
def readRacord(filename):

    reader = tf.TFRecordReader()
    file_queue = tf.train.string_input_producer([filename])
    _ ,raw_image = reader.read(file_queue)
    print("raw_image",raw_image)
    #parse file to image
    features = tf.parse_single_example(raw_image,
                                       features={
                                            "label":tf.FixedLenFeature([],tf.int64),
                                            "image_raw":tf.FixedLenFeature([],tf.string)
                                        })
    image = tf.decode_raw(features["image_raw"],tf.uint8)
    image = tf.reshape(image,[IMAGE_SIZE,IMAGE_SIZE,IMAGE_CHANEL])
    image = tf.cast(image,tf.float32)
    # image = tf.image.random_brightness(image,max_delta=63)
    # image = tf.random_crop(image,[80,80,1])
    # image = tf.image.random_contrast(image,lower=0.2,upper=1.8)
    # image = tf.image.per_image_standardization(image)
    image = image / 255
    label = tf.cast(features["label"],tf.int32)
    return image,label

def createBatch(filename,batch_size):
    image,label = readRacord(filename)
    min_after_deque = 5
    capacity = min_after_deque + batch_size * 2
    image_batch,label_batch = tf.train.shuffle_batch([image,label],
                                                     batch_size=batch_size,
                                                     capacity=capacity,
                                                     min_after_dequeue=min_after_deque
                                             )
    tf.summary.image("images",image_batch,max_outputs=20)
    # label_batch = tf.one_hot(label_batch, depth = 4)
    return image_batch,label_batch