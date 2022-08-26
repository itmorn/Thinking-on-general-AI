import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 1  # 设置最低特征点匹配数量为10
target = cv2.imread('9.png', 0)  # queryImage
template = cv2.imread('10.png', 0)  # trainImage
# target = cv2.imread('../b02_test_sift/6.png', 0)  # queryImage
# template = cv2.imread('../b02_test_sift/6.png', 0)  # trainImage
fast = cv2.FastFeatureDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

kp = fast.detect(template,None)
kp1, des1 = brief.compute(template, kp)

fast = cv2.FastFeatureDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
kp = fast.detect(target,None)
kp2, des2 = brief.compute(target, kp)


# 创建设置FLANN匹配
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1.astype(np.float32), des2.astype(np.float32), k=2)
# store all the good matches as per Lowe's ratio test.
good = []
# 舍弃大于0.7的匹配
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)
print(good)
if len(good) >= MIN_MATCH_COUNT:
    # 获取关键点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    print(src_pts)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    # 计算变换矩阵和MASK
    # 计算多个二维点对之间的最优单映射变换矩阵 H（3行x3列） ，使用最小均方误差或者RANSAC方法
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()  # 先将mask变成一维，再将矩阵转化为列表
    h, w = template.shape
    # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    cv2.polylines(target, [np.int32(dst)], True, 0, 2, cv2.LINE_AA)
else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    matchesMask = None
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=None,
                   matchesMask=matchesMask,
                   flags=2)
result = cv2.drawMatches(template, kp1, target, kp2, good, None, **draw_params)
cv2.imwrite("res.png",result)
plt.imshow(result, 'gray')
plt.show()