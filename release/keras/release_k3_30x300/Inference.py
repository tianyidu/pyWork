import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,load_model
from keras.optimizers import adagrad,adam
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Activation,Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.losses import sparse_categorical_crossentropy
from keras.initializers import TruncatedNormal,VarianceScaling
from keras import regularizers
import Input

NUM_CLASS = 3
model = Sequential()

model.add(Conv2D(32,(3,3),padding="SAME",kernel_initializer=TruncatedNormal(stddev=1e-1,mean=0.001),name="conv1_1",input_shape=(300,300,3)))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
model.add(Conv2D(32,(3,3),padding="SAME",kernel_initializer=TruncatedNormal(stddev=1e-1,mean=0.001),name="conv1_2"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
#model.add(Conv2D(32,(3,3),padding="SAME",kernel_initializer=TruncatedNormal(stddev=1e-1),name="conv1_3"))
#model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="SAME",name="pool1"))

model.add(Conv2D(48,(3,3),padding="SAME",kernel_initializer=TruncatedNormal(stddev=1e-1,mean=0.001),name="conv2_1"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
model.add(Conv2D(48,(3,3),padding="SAME",kernel_initializer=TruncatedNormal(stddev=1e-1,mean=0.001),name="conv2_2"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
#model.add(Conv2D(48,(3,3),padding="SAME",kernel_initializer=TruncatedNormal(stddev=1e-1),name="conv2_3"))
#model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="SAME",name="pool2"))

model.add(Conv2D(64,(3,3),padding="SAME",kernel_initializer=TruncatedNormal(stddev=1e-1,mean=0.001),name="conv3_1"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
model.add(Conv2D(64,(3,3),padding="SAME",kernel_initializer=TruncatedNormal(stddev=1e-1,mean=0.001),name="conv3_2"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
model.add(Conv2D(64,(3,3),padding="SAME",kernel_initializer=TruncatedNormal(stddev=1e-1,mean=0.001),name="conv3_3"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="SAME",name="pool3"))

model.add(Conv2D(96,(3,3),padding="SAME",kernel_initializer=TruncatedNormal(stddev=1e-1,mean=0.001),name="conv4_1"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
model.add(Conv2D(96,(3,3),padding="SAME",kernel_initializer=TruncatedNormal(stddev=1e-1,mean=0.001),name="conv4_2"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
model.add(Conv2D(96,(3,3),padding="SAME",kernel_initializer=TruncatedNormal(stddev=1e-1,mean=0.001),name="conv4_3"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="SAME",name="pool4"))

model.add(Conv2D(128,(3,3),padding="SAME",kernel_initializer=TruncatedNormal(stddev=1e-1,mean=0.001),name="conv5_1"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
model.add(Conv2D(128,(3,3),padding="SAME",kernel_initializer=TruncatedNormal(stddev=1e-1,mean=0.001),name="conv5_2"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
model.add(Conv2D(150,(3,3),padding="SAME",kernel_initializer=TruncatedNormal(stddev=1e-2,mean=0.001),name="conv5_3"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="SAME",name="pool5"))

model.add(Flatten())

model.add(Dense(units=1000,kernel_initializer=TruncatedNormal(stddev=1/1000.0,mean=0),name="local1"))
model.add(Activation("relu"))
model.add(Dropout(0.4))

model.add(Dense(units=1000,kernel_initializer=TruncatedNormal(stddev=1e-3,mean=0),name="local2"))
model.add(Activation("relu"))
model.add(Dropout(0.4))

model.add(Dense(units=512,kernel_initializer=TruncatedNormal(stddev=1/512.0,mean=0),name="local3"))
model.add(Activation("relu"))
model.add(Dropout(0.4))

model.add(Dense(units=NUM_CLASS,kernel_initializer=TruncatedNormal(stddev=1e-1,mean=0),name="local4"))
model.add(Activation("softmax"))

model.summary()
model.compile(loss="sparse_categorical_crossentropy",optimizer=adam(lr=1e-4, decay=0.01),metrics=["accuracy"])

imgInput = Input.ImgInput(r"/root/pworkspace/lj_k3/data")

ckpt = keras.callbacks.ModelCheckpoint(filepath='../ckpt/weights.hdf5',mode='auto',monitor="acc",verbose=1,save_best_only=True)
#model.fit_generator(generator=imgInput,epochs=100,shuffle=True,max_queue_size=5,callbacks=[ckpt])
x,y = imgInput.getOneFile()
#print(x.shape,y.shape,x[0])
#print(y[:20])
model.fit(x=x,y=y,batch_size=9,epochs=100,validation_split=0.1,shuffle=True,callbacks=[ckpt])
#train_data_dir = r"/root/pworkspace/lj_k3/pic3genter/data"
#validation_data_dir = r"/root/pworkspace/lj_k3/pic3genter/test"
#img_width = img_height = 300
# this is the augmentation configuration we will use for training
#train_datagen = ImageDataGenerator(rescale=1./255)

# this is the augmentation configuration we will use for testing:
# only rescaling
#test_datagen = ImageDataGenerator(rescale=1./255)

#train_generator = train_datagen.flow_from_directory(
#        train_data_dir,
#        target_size=(img_width, img_height),
#        batch_size=9,
#        class_mode='binary')

#validation_generator = test_datagen.flow_from_directory(
#        validation_data_dir,
#        target_size=(img_width, img_height),
#        batch_size=9,
#        class_mode='binary')

#model.fit_generator(
#        train_generator,
#        epochs=100,
#        validation_data=validation_generator)

