import numpy as np
import cv2
import time
img = cv2.imread('../a01_test_canny/123.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
a = time.time()
keypoints = sift.detect(img_gray, None)
kp2, des2 = sift.detectAndCompute(img_gray, None)
print(time.time()-a)
img_sift = np.copy(img)

cv2.drawKeypoints(img, keypoints, img_sift, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('res_sift.png', img_sift)