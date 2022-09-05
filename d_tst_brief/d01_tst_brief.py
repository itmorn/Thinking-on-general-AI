import time

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# img = cv.imread('../a01_test_canny/123.jpg',0)
img = cv.imread('../images/12.jpg',0)
# 初始化FAST检测器
fast = cv.FastFeatureDetector_create()
brief = cv.xfeatures2d.BriefDescriptorExtractor_create(16)

a = time.time()
kp = fast.detect(img,None)
print(time.time()-a)

a = time.time()
kp, des = brief.compute(img, kp)
print(time.time()-a)

print( brief.descriptorSize() )
print( des.shape )

img = cv.drawKeypoints(img,kp,img,color=(0,0,255))
cv.imwrite("res.png", img)