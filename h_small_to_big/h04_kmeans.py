# 对特征点进行聚类
import shutil

import os
from sklearn.cluster import KMeans

import numpy as np

if __name__ == '__main__':
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
    print()

    NXY = arr_all[:,:3]
    vec = arr_all[:,3:]
    print("len(vec)", len(vec))

    k_means = KMeans(n_clusters=3, n_jobs=-1)#random_state=10,
    k_means.fit(vec)
    print(k_means.cluster_centers_)
    np.save("k_means",k_means.cluster_centers_)