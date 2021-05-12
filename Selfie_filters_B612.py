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
    '''
    Plot image and facial keypoints
    Parameters:
    --------------------
    img: Image column value
    face_point: Target column value
    '''

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
