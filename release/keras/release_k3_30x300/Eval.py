from keras.models import load_model

import Input
import os
from time import sleep
from datetime import datetime

model_name = "../ckpt/weights.hdf5"

while True:
    if os.path.exists(model_name):
        print("Using model ",model_name)
        model = load_model(model_name)
        imgInput = Input.ImgInput(r"/root/pworkspace/lj_k3/data",partOfFile="eval")
        result = model.evaluate_generator(generator=imgInput)
        result_label = model.metrics_names
        print(datetime.now(),"INFO:  ",result_label[0],":",result[0]," -- ",result_label[1],result[1])
    else:
        print("model file not exists")
    break
    sleep(150)
