"""
@Auth: itmorn
@Date: 2022/8/30-19:55
@Email: 12567148@qq.com
"""
# coding=utf-8
import cv2
import numpy as np

img = cv2.imread("../a01_test_canny/123.jpg", 0)
x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

absX = cv2.convertScaleAbs(x)  # 转回uint8
absY = cv2.convertScaleAbs(y)
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

cv2.imwrite("res.png",dst*2)