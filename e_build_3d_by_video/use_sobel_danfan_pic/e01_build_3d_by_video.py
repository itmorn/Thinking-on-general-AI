# 视频提取特征
import os
import time
import shutil
import cv2
import ffmpeg
import sys
import numpy as np
import json

def orb(img_gray,num_frame,frame):
    # a = time.time()
    kp, des = orb_detetor.detectAndCompute(img_gray, None)
    # print(time.time() - a)
    arr_xy = np.array([[i.pt[0], i.pt[1]] for i in kp],dtype=np.float32)
    des = des.astype(np.float32)
    des = des/((np.sum(des**2,axis=1)**0.5).reshape(-1,1))
    arr_xy_des = np.c_[arr_xy, des]
    np.save(f"npy/{num_frame}", arr_xy_des)

    img_orb = np.copy(frame)
    cv2.drawKeypoints(frame, kp, img_orb, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return img_orb

if __name__ == '__main__':

    shutil.rmtree("pic_frame",ignore_errors=True)
    os.makedirs("pic_frame")

    shutil.rmtree("pic_orb",ignore_errors=True)
    os.makedirs("pic_orb")

    shutil.rmtree("npy",ignore_errors=True)
    os.makedirs("npy")

    orb_detetor = cv2.ORB_create(2000)
    num = 0.0

    dir_in = "danfan2022_08_29/"
    dir_in = "outline/"
    # dir_in = "apple/"
    lst_file = [dir_in + i for i in os.listdir(dir_in)]
    lst_file.sort()
    for url_img in lst_file:
        print(url_img)
        frame = cv2.imread(url_img)

        # frame = cv2.resize(frame, (0, 0), fx=0.05, fy=0.05)

        num_frame = "%06d" % int(num*10)

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x = cv2.Sobel(img_gray, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(img_gray, cv2.CV_16S, 0, 1)

        absX = cv2.convertScaleAbs(x)  # 转回uint8
        absY = cv2.convertScaleAbs(y)
        img_gray = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        img_gray = np.clip(img_gray.astype(np.float32) * 5, 0, 255).astype(np.uint8)
        cv2.imwrite(f"pic_frame/{num_frame}.png", img_gray)
        img_orb = orb(img_gray,num_frame,frame)
        cv2.imwrite(f"pic_orb/{num_frame}.png", img_orb)
        print(num,len(lst_file))
        num+=1
