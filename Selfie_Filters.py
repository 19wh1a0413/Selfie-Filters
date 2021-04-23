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

def put_hat(hat, fc, x, y, w, h):
    face_width = w
    face_height = h

    hat_width = face_width + 1
    hat_height = int (0.50 * face_height) + 1

    hat = cv2.resize(hat, (hat_width, hat_height))

    for i in range (hat_height):
        for j in range (hat_width):
            for k in range (3):
                if hat[i][j][k] < 235:
                    fc[y + i - int (0.40 * face_height)][x + j][k] = hat[i][j][k]
    return fc

def put_mask(mask, fc, x, y, w, h):
    face_width = w
    face_height = h

    ears_width =  face_width + 10
    ears_height = int(0.7 * face_height) + 1

    mask = cv2.resize(mask, (ears_width, ears_height))

    for i in range(ears_height):
        for j in range(ears_width):
            for k in range(3):
                if mask[i][j][k] < 235:
                    fc[y + i - int(-0.40 * face_height)][x + j][k] = mask[i][j][k]
    return fc


def put_crown(crown, fc, x, y, w, h):
    face_width = w
    face_height = h

    crown_width = face_width + 10
    crown_height = int (0.75 * face_height) + 10

    crown = cv2.resize(crown, (crown_width, crown_height))

    for i in range (crown_height):
        for j in range (crown_width):
            for k in range (3):
                if crown[i][j][k] < 235:
                    fc[y + i - int (0.50 * face_height)][x + j][k] = crown[i][j][k]
    return fc
def put_ears(ears, fc, x, y, w, h):
    face_width = w
    face_height = h

    ears_width = face_width + 1
    ears_height = int (1 * face_height) + 1

    ears = cv2.resize(ears, (ears_width, ears_height))

    for i in range (ears_height):
        for j in range (ears_width):
            for k in range (3):
                if ears[i][j][k] < 233:
                    fc[y + i - int (0.55 * face_height)][x + j][k] = ears[i][j][k]
    return fc


global choise
choice = 0
print ('Enter your choice filter to launch :')
print('1 = hat & glasses\n'
      '2 = Butterflies\n'
      '3 = ears and mask\n'
      '4 = dog\n'
      '5 = crown')
choise = int (input ('enter your choice:'))
webcam = cv2.VideoCapture (0)
out = cv2.VideoWriter(filename, get_video_type(filename),25,(640,480))
img_counter = 0

while True:
    size = 4
    ret, im = webcam.read ()
    im = cv2.flip (im, 1, 0)
    gray = cv2.cvtColor (im, cv2.COLOR_BGR2GRAY)
    fl = face.detectMultiScale(gray, 1.20, 8)


    for (x, y, w, h) in fl:
        if choise == 1:
            im = put_hat(hat, im, x, y, w, h)
            im = put_glass(glass, im, x, y, w, h)

        elif choise == 2:
            im = put_butterflies(butterflies, im, x, y, w, h)
        elif choise == 3:
            im = put_ears(ears, im, x, y, w, h)
            im = put_mask(mask, im, x, y, w, h)
        elif choise == 4:
            im = put_dog_filter(dog, im, x, y, w, h)

        else:
            im = put_crown(crown, im, x, y, w, h)

    out.write(im)
    cv2.imshow('OUTPUT', im)
    key = cv2.waitKey(1) & 0xff
    if key == 27:  # The Esc key
        break
    if key % 256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, im)
        print("{} written!".format(img_name))
        img_counter += 1


webcam.release()
out.release()
cv2.destroyAllWindows()
