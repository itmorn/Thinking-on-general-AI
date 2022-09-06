"""
@Auth: itmorn
@Date: 2022/9/5-17:56
@Email: 12567148@qq.com
"""
import cv2
img = cv2.imread("../images/idcard.png", 0)
# img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)

edges = cv2.Canny(img,50,150)

cv2.imwrite("res_canny.png", edges)