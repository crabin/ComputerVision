from __future__ import print_function
import cv2 as cv
import argparse
 
'''
该代码尝试使用背景差分法，完成了固定摄像头中，动态物体的提取。
'''
#有两种算法可选，KNN和MOG2，下面的代码使用KNN作为尝试
algo='KNN'
if algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()
#打开一个视频文件
capture = cv.VideoCapture(cv.samples.findFileOrKeep('video/Camera Road 02.avi'))
#判断视频是否读取成功
if not capture.isOpened():
    print('Unable to open')
    exit(0)

frame_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(capture.get(cv.CAP_PROP_FPS))

fourcc = cv.VideoWriter_fourcc(*'XVID')
fourcc1 = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))
out1 = cv.VideoWriter('output1.avi', fourcc1, fps, (frame_width, frame_height))

#逐帧读取视频，进行相关分析
while True:
    #读取视频的第一帧
    ret, frame = capture.read()
    if frame is None:
        break
    #使用定义的backSub对象，输入新的一帧frame，生成背景蒙版
    fgMask = backSub.apply(frame)
    #将原视频的当前帧和蒙版做相加运算，将前景物体提取出来
    Object=cv.add(frame,frame,mask=fgMask)

    out.write(Object)
    out1.write(fgMask)

    #展示视频中的物体，三个窗口分别表示原视频、背景、移动目标
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    cv.imshow('Object',Object)
    #每帧展示结束，等待30毫秒
    keyboard = cv.waitKey(30)
    #按q推出程序
    if keyboard == 'q' or keyboard == 27:
        break