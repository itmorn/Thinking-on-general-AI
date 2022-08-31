# 计算每个特征点 到 最近聚类中心的距离  用颜色深浅 表示匹配度
import cv2
import shutil

import os
import numpy as np

def orb(frame):

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # a = time.time()
    kp, arr_des = orb_detetor.detectAndCompute(img_gray, None)
    # print(time.time() - a)
    arr_xy = np.array([[i.pt[0], i.pt[1]] for i in kp],dtype=np.int32)

    img_orb = np.copy(frame)
    cv2.drawKeypoints(frame, kp, img_orb, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite("res.png",img_orb)

    return arr_xy,arr_des

if __name__ == '__main__':
    orb_detetor = cv2.ORB_create(100000)
    # img = cv2.imread("../b02_test_sift/4ed7a3e4962b59bbfea5c6d959bdff1.jpg",1)
    img = cv2.imread("../a01_test_canny/12.jpg",1)
    H,W,_ = img.shape
    arr_xy,arr_des = orb(img)

    arr_des = arr_des.astype(np.float32)
    arr_des = arr_des / ((np.sum(arr_des ** 2, axis=1) ** 0.5).reshape(-1, 1))
    arr_kmeans = np.load("k_means.npy")
    arr_res = np.max(np.matmul(arr_des, arr_kmeans.T), axis=1)

    for idx_pic,max_val in enumerate(arr_res):
        if max_val<0.9:
            continue
        X,Y = arr_xy[idx_pic]
        des = arr_des[idx_pic]

        img[max(Y-5,0):min(Y+5,H), max(X-5,0):min(X+5,W), 0] = 0
        img[max(Y-5,0):min(Y+5,H), max(X-5,0):min(X+5,W), 1] = 0
        img[max(Y-5,0):min(Y+5,H), max(X-5,0):min(X+5,W), 2] = min(255,255*max_val)
        print(idx_pic)
    cv2.imwrite("res.png",img)
