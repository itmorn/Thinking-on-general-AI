# 计算每个特征点 到 最近聚类中心的距离  用颜色深浅 表示匹配度
import cv2
import shutil

import os
import numpy as np

def orb(img_gray):

    # a = time.time()
    kp, arr_des = orb_detetor.detectAndCompute(img_gray, None)
    # print(time.time() - a)
    arr_xy = np.array([[i.pt[0], i.pt[1]] for i in kp],dtype=np.int32)

    return arr_xy,arr_des

if __name__ == '__main__':
    orb_detetor = cv2.ORB_create(100000)
    # img = cv2.imread("../b02_test_sift/4ed7a3e4962b59bbfea5c6d959bdff1.jpg",1)
    img = cv2.imread("../../images/12.jpg", 1)

    img = cv2.resize(img, (0, 0), fx=0.08, fy=0.08)


    H,W,_ = img.shape

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = cv2.Sobel(img_gray, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img_gray, cv2.CV_16S, 0, 1)

    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    img_gray = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    img_gray = np.clip(img_gray.astype(np.float32) * 5, 0, 255).astype(np.uint8)

    arr_xy,arr_des = orb(img_gray)

    arr_des = arr_des.astype(np.float32)
    arr_des = arr_des / ((np.sum(arr_des ** 2, axis=1) ** 0.5).reshape(-1, 1))
    arr_kmeans = np.load("k_means.npy")
    arr_kmeans = arr_kmeans / ((np.sum(arr_kmeans ** 2, axis=1) ** 0.5).reshape(-1, 1))


    arr_res = np.max(np.matmul(arr_des, arr_kmeans.T), axis=1)

    for idx_pic,max_val in enumerate(arr_res):
        if max_val<0.95:
            continue
        X,Y = arr_xy[idx_pic]
        des = arr_des[idx_pic]

        img[max(Y-1,0):min(Y+1,H), max(X-1,0):min(X+1,W), 0] = 0
        img[max(Y-1,0):min(Y+1,H), max(X-1,0):min(X+1,W), 1] = 0
        img[max(Y-1,0):min(Y+1,H), max(X-1,0):min(X+1,W), 2] = min(255,255*max_val)
        print(idx_pic)
    cv2.imwrite("img_gray.png",img_gray)
    cv2.imwrite("res.png",img)
