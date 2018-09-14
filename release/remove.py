import cv2
import numpy as np


file = r"E:\PWORKSPACE\houseUtil\house\bedroom\test-1c752b2a-8e24-425f-8a66-6cd93f9f1db1.jpg"

if __name__ == "__main__":
    img = cv2.imread(file)
    # print(img,type(img),img.shape,dir(img))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # _, img = cv2.threshold(img, 177, 255, cv2.THRESH_BINARY)
    # img = cv2.medianBlur(img,15)
    # img = cv2.GaussianBlur(img,(0,0),1)
    # img = cv2.blur(img,(15,15))
    # img = cv2.bilateralFilter(img,75,75,5)

    print(img,img.shape)
    # np.savetxt("f:/tem.text",img)
    cv2.imshow("img",img)
    cv2.waitKey(0)