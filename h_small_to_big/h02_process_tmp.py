import os

import cv2
import numpy as np

if __name__ == '__main__':

    # orb = cv2.ORB_create(100)
    sift = cv2.SIFT_create()

    dir_in = "../images/outline/"
    lst_file = [dir_in + i for i in os.listdir(dir_in)]
    for idx,url_img in enumerate(lst_file):
        img = cv2.imread(url_img,1)
        img = cv2.resize(img, (0, 0), fx=0.02, fy=0.02)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        print(img_gray.shape)

        keypoints, des2 = sift.detectAndCompute(img_gray, None)

        img_detect = np.copy(img)

        cv2.drawKeypoints(img, keypoints, img_detect, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(f'res_orb{idx}.png', img_detect)

