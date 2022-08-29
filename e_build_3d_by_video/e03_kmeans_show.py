# 展示距离聚类中心最近的图像区域
import cv2
import shutil

import os
import numpy as np

if __name__ == '__main__':

    shutil.rmtree("kmeans_show",ignore_errors=True)
    os.makedirs("kmeans_show")

    dir_in = "npy/"
    lst_npy = [dir_in + i for i in os.listdir(dir_in)]
    lst_npy.sort()
    arr = np.load(lst_npy[0])
    arr_all = np.c_[np.ones(len(arr))*int(lst_npy[0].split("/")[-1].split(".")[0]), arr]
    for npy in lst_npy[1:]:
        print(npy)
        arr= np.load(npy)
        arr = np.c_[np.ones(len(arr))*int(npy.split("/")[-1].split(".")[0]), arr]
        arr_all = np.r_[arr_all,arr]

    NXY = arr_all[:,:3]
    vec = arr_all[:,3:]


    arr_kmeans = np.load("k_means.npy")
    for center in arr_kmeans:
        arr_dis = np.sum((vec - center) ** 2, axis=1) ** 0.5
        idx_min = np.argmin(arr_dis)
        N,X,Y = [int(i) for i in NXY[idx_min]]
        img = cv2.imread(f"pic_frame/{'%06d' % N}.png")
        img[Y, X, 0] = 0
        img[Y, X, 1] = 0
        img[Y, X, 2] = 255
        cv2.imwrite(f"kmeans_show/{'%06d' % N}.png",img)
        print()