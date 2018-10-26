import logging
import os
import time

# 1、logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

# s = ["I come to China to travel",
#     "This is a car polupar in China",
#     "I love tea and Apple ",
#     "The work is to write some papers in science"]
# t = " ".join(s)
# logging.info(t.split(" "))
#
# logging.info(set(t.split(" ")))
# logging.info(len(set(t.split(" "))))

# 2、with
# class testWith:
#     def __enter__(self):
#         logging.info("__enter__")
#         return self
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         logging.info("__exit__")
#         logging.info(exc_type)
#         logging.info(exc_val)
#         logging.info(exc_tb)
#         logging.info("*__exit__*")
#
#     def test(self):
#         return "test with"
#
# with testWith() as t:
#     logging.info("******"+t.test())

#3、sqlite
# import sqlite3
# cur = sqlite3.connect(r"E:\PWORKSPACE\house3\db.sqlite3")
# print(cur.execute(".tables"))

#4、多进程
# import multiprocessing
#
# def wait(t=3):
#     logging.info(str(os.getpid())+" waitting ..... ")
#     time.sleep(t)
#     logging.info(str(os.getpid())+" end ")
#
# if __name__ == "__main__":
#     l = []
#     for i in range(5):
#         p = multiprocessing.Process(target=wait)
#         l.append(p)
#         p.start()
#     for i in l:
#         i.join()
#     print("end")

# 进程池
import multiprocessing
def teste(i):
    print("stept into ",i)
    return "hello "+str(i)

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=3)
    result = []
    for i in range(8):
        # print(i)
        result.append(pool.apply_async(func=teste,args=(i,)))
    pool.close()
    pool.join()
    for re in result:
        print(re.get())


#5、乱序
# import random
# all_companys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 18, 19, 20, 21, 22, 24, 25]
# random.shuffle(all_companys)
# logging.info(all_companys)

#6、pandas
# a=[[123,"asdlfkj","kjkj","dd"],
#     [456,"asdlfkj","kjkj","ee"],
#     [789, "asdlfkj", "kjkj","ff"]]
# import pandas as pd
# df = pd.DataFrame(a,columns=["a","b","c","d"])
# print(df[df["a"]==123].shape,df[(df["a"]==123)|(df["a"]==456)].shape)
# print(df.shape)
# print(df.count())