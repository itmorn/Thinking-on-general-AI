"""
对模板多尺度建模
对待测图像从小到大匹配模板，不断确认和调整

@Auth: itmorn
@Date: 2022/8/31-9:51
@Email: 12567148@qq.com
"""
import cv2
import numpy as np

if __name__ == '__main__':

    img = cv2.imread("../images/123.jpg",1)
    img = cv2.resize(img, (0, 0), fx=0.03, fy=0.03)


    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(img_gray.shape)

    cv2.imwrite("img_gray.png",img_gray)

    # orb = cv2.ORB_create(100)
    sift = cv2.SIFT_create()

    keypoints, des2 = sift.detectAndCompute(img_gray, None)

    img_detect = np.copy(img)

    cv2.drawKeypoints(img, keypoints, img_detect, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('res_orb.png', img_detect)

