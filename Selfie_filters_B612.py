import numpy as np
import pandas as pd
import cv2, os
from cnn_model import *
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from keras import layers, callbacks, utils, applications, optimizers
from keras.models import Sequential, Model, load_model


frames_per_second = 25.0

def prepare_data(df):
    '''
    Prepare data (image and target variables) for training
    '''
    df['Image'] = df['Image'].apply(lambda x: np.fromstring(x, sep=' '))

    df = df.dropna()

    X = np.vstack(df['Image'].values) / 255.
    X = X.astype(np.float32)
    X = X.reshape(-1, 96,96, 1)

    y = df[df.columns[:-1]].values #(30 columns)
    y = (y - 48) / 48
    X, y = shuffle(X, y, random_state=42)
    y = y.astype(np.float32)

    return X,y


def plot_data(img, face_points):
    fig = plt.figure (figsize=(30, 30))
    ax = fig.add_subplot (121)

    ax.imshow (np.squeeze (img), cmap='gray')
    face_points = face_points * 48 + 48

    ax.scatter (face_points[0::2], face_points[1::2], marker='o', c='blue', s=25)
    plt.show ()

df = pd.read_csv('training.csv')
df=df.head(10)

X_train, y_train =prepare_data(df)

plot_data(X_train[3], y_train[3])


my_model = create_model()
compile_model(my_model, optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

save_model(my_model, 'models/mm')

def apply_filters(face_points, image_copy_1, image_name):
    animal_filter = cv2.imread ("images/" + image_name, cv2.IMREAD_UNCHANGED)

    for i in range (len (face_points)):
        filter_width = 7 * (face_points[i][14] + 15 - face_points[i][18] + 15)
        scale_factor = filter_width / animal_filter.shape[1]
        sg = cv2.resize (animal_filter, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

        width = sg.shape[1]
        height = sg.shape[0]

        x1 = int ((face_points[i][2] + 5 + face_points[i][0] + 5) / 2 - width / 2)
        x2 = x1 + width

        y1 = int ((face_points[i][2] - 65 + face_points[i][1] - 65) / 2 - height / 2)
        y2 = y1 + height

        alpha_fil = np.expand_dims (sg[:, :, 3] / 255.0, axis=-1)
        alpha_face =1.0 - alpha_fil

        image_copy_1[y1:y2, x1:x2] = (alpha_fil * sg[:, :, :3] + alpha_face * image_copy_1[y1:y2, x1:x2])

    return image_copy_1

model = load_model('models/mm.h5')


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

global choice
print ('Enter your choice filter to launch :')
print('1 = Rabbit\n'
      '2 = Dog\n'
      '3 = Pig\n'
      '4 = Fluffy\n'
      '5 = Mask\n'
      '6 = Bear\n')
choice = int (input ('enter your choice:'))

camera = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('video.mp4', fourcc,25.0,(640,480))
img_counter = 0
while True:
    
    ret, image = camera.read ()
    image_copy = np.copy (image)
    image_copy_1 = np.copy (image)
    image_copy_2 = np.copy (image)

    
    gray = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces_keypoints = []

for (x, y, w, h) in faces:

        
        face = gray[y:y + h, x:x + w]

        
        scaled_face = cv2.resize (face, (96, 96), 0, 0, interpolation=cv2.INTER_AREA)

       
        input_image = scaled_face / 255

     
        input_image = np.expand_dims (input_image, axis=0)
        input_image = np.expand_dims (input_image, axis=-1)

        
        face_points = model.predict (input_image)[0]

        
        face_points[0::2] = face_points[0::2] * w / 2 + w / 2 + x
        face_points[1::2] = face_points[1::2] * h / 2 + h / 2 + y
        faces_keypoints.append (face_points)

        for point in range (15):
            cv2.circle (image_copy, (face_points[2 * point], face_points[2 * point + 1]), 2, (255, 255, 0), -1)

        for (x, y, w, h) in faces:
            if choice == 1:
                s = apply_filters (faces_keypoints, image_copy_1, "s.png")
                out.write (image_copy_1)
                cv2.imshow ('Screen with filter', s)
            elif choice == 2 :
                dog = apply_filters(faces_keypoints, image_copy_1, "dog.png")
                out.write(image_copy_1)
                cv2.imshow('Screen with filter', dog)
            elif choice == 3 :
                pig = apply_filters(faces_keypoints, image_copy_1, "pig.png")
                out.write(image_copy_1)
                cv2.imshow('Screen with filter', pig)
            elif choice == 4:
                fluffy = apply_filters(faces_keypoints, image_copy_1, "fluffy.png")
                out.write(image_copy_1)
                cv2.imshow('Screen with filter', fluffy)
            elif choice == 5:
                mask = apply_filters (faces_keypoints, image_copy_1, "mask.png")
                out.write (image_copy_1)
                cv2.imshow ('Screen with filter', mask)
            else:
                bear = apply_filters (faces_keypoints, image_copy_1, "bear.png")
                out.write (image_copy_1)
                cv2.imshow ('Screen with filter', bear)

        cv2.imshow ('Screen with facial Keypoints predicted', image_copy)
    key = cv2.waitKey(1) & 0xff
    if key == 27:  
        break
    if key % 256 == 32:
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, image_copy_1)
        print("{} saved!!!".format(img_name))
        img_counter += 1

camera.release()
out.release()
cv2.destroyAllWindows()    
