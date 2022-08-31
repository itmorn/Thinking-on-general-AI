# 视频提取特征
import os
import time
import shutil
import cv2
import ffmpeg
import sys
import numpy as np
import json

def read_frame_by_time(in_file, time):
    """
    指定时间节点读取任意帧
    """
    out, err = (
        ffmpeg.input(in_file, ss=time)
            .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
            .run(capture_stdout=True, quiet=True)
    )
    return out


def get_video_info(in_file):
    """
    获取视频基本信息
    """
    try:
        probe = ffmpeg.probe(in_file)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream is None:
            print('No video stream found', file=sys.stderr)
            sys.exit(1)
        return video_stream
    except ffmpeg.Error as err:
        print(str(err.stderr, encoding='utf8'))
        sys.exit(1)


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

    orb_detetor = cv2.ORB_create(1000)
    VIDEO_NAME = "../dde38028db3d9b60269685687623736d.mp4"

    DURATION_TIME = 0.2  # 每隔x秒提取一张图片

    video_info = get_video_info(VIDEO_NAME)
    total_duration = float(video_info['duration'])
    num = 0.0
    while num<total_duration:
        frame_b = read_frame_by_time(VIDEO_NAME, num)
        frame = cv2.imdecode(np.asarray(bytearray(frame_b), dtype="uint8"), cv2.IMREAD_COLOR)
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
        print(num,total_duration)
        num+=DURATION_TIME
