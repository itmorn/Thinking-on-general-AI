"""
@Auth: itmorn
@Date: 2022/8/25-22:00
@Email: 12567148@qq.com
"""
# 导包
import cv2
from matplotlib import pyplot as plt
# 选一个实验用图
# img = cv2.imread('123.jpg',0)
img = cv2.imread('2018828181532_5uy8L.jpeg',0)
# 朋友脸大，给他缩小点
# img = cv2.resize(img,(800,800))
# 来直接干Canny，设两个阈值100，200
# edges = cv2.Canny(img,50,150)
# cv.imwrite("res.png",edges)

# 创建窗口
#定义回调函数
def nothing(x):
    pass

cv2.namedWindow('Canny')

# 创建滑动条，分别对应Canny的两个阈值
cv2.createTrackbar('threshold1', 'Canny', 0, 255, nothing)
cv2.createTrackbar('threshold2', 'Canny', 0, 255, nothing)

while (1):

    # 返回当前阈值
    threshold1 = cv2.getTrackbarPos('threshold1', 'Canny')
    threshold2 = cv2.getTrackbarPos('threshold2', 'Canny')

    img_output = cv2.Canny(img, threshold1, threshold2)

    # 显示图片
    # cv2.imshow('original', img)
    cv2.imshow('Canny', img_output)
    cv2.imwrite("res.png", img_output)
    # 空格跳出
    if cv2.waitKey(1) == ord(' '):
        break

    # 摧毁所有窗口
cv2.destroyAllWindows()