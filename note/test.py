import os
import multiprocessing
import time
import datetime
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar','foo', 'bar', 'foo', 'foo'],
                   'B' : ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
                   'C' : np.random.randn(8),
                   'D' : np.random.randn(8)})
print(df)
grouped = df.groupby(['A']).count()
print(grouped.T.head())

# data = pd.DataFrame({"a":[1,2,1],"b":[4,5,6],"c":[100,800,900]})
# print(data)
# print(data.ix[:,["a","c"]])
# print(data.ix[:,[0,2]])
# print("*"*10)
# print(np.array(data[:1]))
# print(np.array(data[2:]))
# print(np.array(data[:1])==np.array(data[2:]))
# print(np.mean(np.array(data[:1])==np.array(data[2:])))
# print("*"*10)
# std = StandardScaler()
# print(std.fit_transform(data))


# image = Image.open(r"E:\PWORKSPACE\WiwjPicture\frontimgs\rent\1\25703156_2_20180927_inbennkp255f9.jpg")
# print(image.info,image.format in ["PNG","JPEG"])
# print(dir(image))
# print(image.size[0]>224)
#
# print(os.path.getsize(r"F:\tmp\1\41532652_1_20181025_nhageoki849791b1.jpg")<=1000)
# print(os.path.isfile(r"F:\tmp\1\41532652_1_20181025_nhageoki849791b1.jpg"))

# now = datetime.datetime.now()
# print(now - datetime.timedelta(days=1,hours=now.hour,minutes=now.minute,seconds=now.second,microseconds=now.microsecond), now)
#
# s="https://aihome.aihome365.cn/2018/07/06d500f6-50ea-4111-9256-4532b255e818.JPG?x-oss-process=style/aihome"
# print(s.lower().endswith("aihome"),s.lower())
# print(s.split("?")[0])
#rename file
# if __name__ == "__main__":
#     path = r"F:\tmp\董浩测试图片"
#     for subdir in os.listdir(path):
#         subpath = os.path.join(path,subdir)
#         if os.path.isdir(subpath):
#             for img in os.listdir(subpath):
#                 img = os.path.join(subpath,img)
#                 if os.path.isfile(img):
#                     print("----")
#                     name = img.split("_")
#                     if len(name) < 4:
#                         name = name[0]+"_1_"+"_".join(name[1:])
#                         os.rename(img,name)
#                         print("rename file %s to %s " % (img,name))

# s = "http://asdf/asdf/asdf/asdf.jpg"
# if "http" == s[:4]:
#     print("yes",s[:4])
# else:
#     print("no")
#
# def wait(interval):
#     print(os.getpid(),"start waiting")
#     time.sleep(interval)
#     print(os.getpid(),"wait>>>")
#
# if __name__ == "__main__":
#
#     for i in range(2):
#         p = multiprocessing.Process(target=wait,args=(10,))
#         p.start()
#
#     print("okokokokokokok")
#     print("okokok*****okokokok")