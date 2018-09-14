import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.optimizers import adagrad,RMSprop,adam
from keras.initializers import TruncatedNormal
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Activation,Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
from keras import regularizers
from release.k_model import Input
from keras import callbacks
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K
from matplotlib import pyplot as plt
import os
import numpy as np
import uuid

NUM_CLASS = 4
model_name = 'weights.ckpt'
if os.path.exists(model_name):
    model = load_model(model_name)
else:
    model = Sequential()
    layer = Conv2D(32,(3,3),padding="SAME",name="conv1_1",kernel_initializer="glorot_normal",kernel_regularizer=regularizers.l2(),input_shape=(300,300,3))
    # model.add(BatchNormalization(axis=1))
    model.add(layer)
    model.add(LeakyReLU())
    model.add(Conv2D(32,(3,3),padding="SAME",kernel_initializer="glorot_normal",name="conv1_2"))
    # model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="SAME",name="pool1"))

    model.add(Conv2D(48,(3,3),padding="SAME",kernel_initializer="glorot_normal",kernel_regularizer=regularizers.l2(),name="conv2_1"))
    # model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU())
    model.add(Conv2D(48,(3,3),padding="SAME",kernel_initializer="glorot_normal",name="conv2_2"))
    # model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="SAME",name="pool2"))

    model.add(Conv2D(64,(3,3),padding="SAME",kernel_initializer="glorot_normal",kernel_regularizer=regularizers.l2(),name="conv3_1"))
    # model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU())
    model.add(Conv2D(64,(3,3),padding="SAME",kernel_initializer="glorot_normal",name="conv3_2"))
    # model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU())
    model.add(Conv2D(64,(3,3),padding="SAME",kernel_initializer="glorot_normal",name="conv3_3"))
    # model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="SAME",name="pool3"))

    model.add(Conv2D(96,(3,3),padding="SAME",kernel_initializer="glorot_normal",kernel_regularizer=regularizers.l2(),name="conv4_1"))
    # model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU())
    model.add(Conv2D(96,(3,3),padding="SAME",kernel_initializer="glorot_normal",name="conv4_2"))
    # model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU())
    model.add(Conv2D(96,(3,3),padding="SAME",kernel_initializer="glorot_normal",name="conv4_3"))
    # model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="SAME",name="pool4"))

    model.add(Conv2D(128,(3,3),padding="SAME",kernel_initializer="glorot_normal",kernel_regularizer=regularizers.l2(),name="conv5_1"))
    # model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU())
    model.add(Conv2D(128,(3,3),padding="SAME",kernel_initializer="glorot_normal",name="conv5_2"))
    # model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU())
    model.add(Conv2D(128,(3,3),padding="SAME",kernel_initializer="glorot_normal",name="conv5_3"))
    # model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="SAME",name="pool5"))

    model.add(Flatten())

    model.add(Dense(units=512,kernel_initializer="glorot_normal",name="local1"))
    # model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU())
    #model.add(Dropout(0.5))

    model.add(Dense(units=1000,kernel_initializer="glorot_normal",name="local2"))
    # model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.4))

    model.add(Dense(units=1000,kernel_initializer="glorot_normal",name="local3"))
    # model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.4))

    model.add(Dense(units=NUM_CLASS,kernel_initializer="glorot_normal",name="local4"))
    model.add(Activation("softmax"))

    model.summary()
    # model.compile(loss="sparse_categorical_crossentropy",optimizer=adagrad(lr=0.001, decay=0.01),metrics=["acc"])
    model.compile(loss="sparse_categorical_crossentropy",optimizer=adam(lr=0.001, decay=0.01),metrics=["acc"])
    # print("weights",layer.get_weights())

imgInput = Input.ImgInput(r"E:\PWORKSPACE\house3\data3")
ckpt = keras.callbacks.ModelCheckpoint(filepath=model_name,mode='auto' ,save_best_only='False')


# model.fit_generator(generator=imgInput,epochs=5,steps_per_epoch=7,callbacks=[ckpt])
# model.fit_generator(generator=imgInput,epochs=5,steps_per_epoch=1)
# x,y = imgInput.getOneFile()
#
# class getWeight(keras.callbacks.Callback):
#     def on_batch_end(self, batch, logs={}):
#         # print("batch",type(batch),dir(batch))
#         # layer = model.get_layer("local3")
#         # dense1_layer_model = Model(inputs=model.input, outputs=layer.output)
#         # t = np.reshape(x[0],[1,x[0].shape[0],x[0].shape[1],x[0].shape[2]])
#         # out = dense1_layer_model.predict(t)
#         # print("out",out[0],out.shape)
#         # print("weight",layer.get_weights())
#         # plt.imshow(out[0])
#         # plt.show()
#         for i in range(5):
#             plt.imsave(os.path.join(r"E:\PWORKSPACE\house3\data3\tmp", str(batch)+"_"+str(y[i+batch*5]) + "_" + str(uuid.uuid1()) + ".jpg"), x[i+batch*5])
#
model.fit(x=x,y=y,batch_size=5,epochs=1,validation_split=0.5,callbacks=[ckpt])

train_data_dir = r"/home/app_user_5i5j/workspace/pwork/lj_k3/pic3genter/data"
validation_data_dir = r"/home/app_user_5i5j/workspace/pwork/lj_k3/pic3genter/test"
img_width = img_height = 300
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1./255)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        epoch=100,
        validation_data=validation_generator)