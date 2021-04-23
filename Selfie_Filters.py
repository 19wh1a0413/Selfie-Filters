from typing import Any
import numpy as np
import os
import cv2

face = cv2.CascadeClassifier (cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
hat = cv2.imread (r'C:\Users\Admin\PycharmProjects\OpencvPython\scr\hat.png',1)
glass = cv2.imread (r'C:\Users\Admin\PycharmProjects\OpencvPython\scr\glasses.png',1)
dog = cv2.imread (r'C:\Users\Admin\PycharmProjects\OpencvPython\scr\dog.png',1)
butterflies=cv2.imread(r'C:\Users\Admin\PycharmProjects\OpencvPython\scr\butterfly.jpeg',1)
ears = cv2.imread(r'C:\Users\Admin\PycharmProjects\OpencvPython\scr\ears.png',1)
mask = cv2.imread(r'C:\Users\Admin\PycharmProjects\OpencvPython\scr\mask.png',1)
crown = cv2.imread(r'C:\Users\Admin\PycharmProjects\OpencvPython\scr\crown.png',1)


filename = 'video.avi'
frames_per_second = 25.0

# grab resolution dimensions and set video capture to it.
VIDEO_TYPE = {
        'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
        'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}
def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']

def put_dog_filter(dog, fc, x, y, w, h):
    face_width = w
    face_height = h
    dog = cv2.resize(dog, (int(face_width * 1.5), int(face_height * 1.95)))
    for i in range(int(face_height * 1.75)):
        for j in range(int(face_width * 1.5)):
            for k in range(3):
                if dog[i][j][k] < 235:
                    fc[y + i - int(0.375 * h) - 1][x + j - int(0.35 * w)][k] = dog[i][j][k]
    return fc
