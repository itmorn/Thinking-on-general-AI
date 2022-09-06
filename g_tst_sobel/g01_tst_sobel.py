"""
@Auth: itmorn
@Date: 2022/8/30-19:55
@Email: 12567148@qq.com
"""
# coding=utf-8
import cv2
import numpy as np

img = cv2.imread("../images/idcard.png", 0)
img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
# cv2.imwrite("res.png",img)

x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

absX = cv2.convertScaleAbs(x)  # 转回uint8
absY = cv2.convertScaleAbs(y)
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

cv2.imwrite("res.png",dst)