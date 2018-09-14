# import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.utils import Sequence
from keras.models import load_model
import keras.backend as K
import numpy as np
import random
from matplotlib import pyplot as plt

plt.ion()
x_datas = np.arange(-20,20,1)
labels_tmp = x_datas  + 2
labels = []
for k,v in enumerate(labels_tmp):
    labels.append(v + random.random())

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x_datas, labels, 'r+')
plt.show()
plt.pause(0.05)

datas = {"data":x_datas,"label":labels}
# # plt.plot(datas,labels)
# # plt.show()
# # print(datas.shape)
# class data_gene(Sequence):
#     def __init__(self, datasets, batch_size):
#         # print("datasets",datasets)
#         self.datasets = datasets
#         self.batch_size = batch_size
#
#     def __len__(self):
#         # print("len",int(np.ceil(len(self.datasets["data"]) / float(self.batch_size))))
#         return int(np.ceil(len(self.datasets["data"]) / float(self.batch_size)))
#
#     def __getitem__(self, idx):
#         data = self.datasets["data"]
#         label = self.datasets["label"]
#         # print("data",data[1:2])
#         # print("idx",idx)
#         index = idx * self.batch_size
#         # print("batch",data[index:index+self.batch_size],label[index:index+self.batch_size])
#
#         return  data[index:index+self.batch_size],label[index:index+self.batch_size]
#
# color = ["blue","green","gray","yellow","purple"]
# class cbdef(keras.callbacks.Callback):
#     def __init__(self,model):
#         self.model = model
#         self.lines = None
#     def on_epoch_end(self, epoch, logs=None):
#         self.paint(epoch)
#     def paint(self,epoch):
#         # print("epoch ",epoch)
#         if epoch % 2 == 0:
#             # print("paint")
#             y_pre = model.predict(x_datas)
#             # print(self.lines,ax.lines)
#             if self.lines and self.lines[0] in ax.lines:
#                 ax.lines.remove(self.lines[0])
#             self.lines = ax.plot(x_datas, y_pre, "b")
#             plt.show()
#             plt.pause(0.05)
#
# batch_size = 1
# model = Sequential()
# model.add(Dense(units=1,input_dim=1))
# # model.add(Activation("relu"))
#
def accdef(y,y_pre):
    y_pre  = np.reshape(y_pre,[-1])[0]

    result = np.abs(y-y_pre)<=0.5
    print(result)
    return K.mean(result)

# opt = keras.optimizers.Adagrad(lr=10, epsilon=1e-06)
# model.compile(loss="mse",optimizer=opt,metrics=["acc",accdef])
# # model_json = model.to_json()
# # with open("model,json") as f:
# #     f.write(model_json)
#
# cpkt = keras.callbacks.ModelCheckpoint(filepath='weights.ckpt',monitor='accdef',mode='auto' ,save_best_only='True')
# model.fit_generator(generator=data_gene(datas,batch_size),epochs=10,max_queue_size=5,shuffle=False,callbacks=[cbdef(model),cpkt])
# y_pre = model.predict(datas["data"])

model = load_model("weights.ckpt",custom_objects={"accdef":accdef})
print(model.predict([19]))
# [19.683851]
# [20.697102]
# [21.710352]]