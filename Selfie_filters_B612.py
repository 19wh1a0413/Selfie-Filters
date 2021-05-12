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
    # Create numpy array for pixel values in image column that are separated by space
    df['Image'] = df['Image'].apply(lambda x: np.fromstring(x, sep=' '))

    # Drop all rows that have missing values in them
    df = df.dropna()

    # Normalize the pixel values, scale values between 0 and 1
    X = np.vstack(df['Image'].values) / 255.
    X = X.astype(np.float32)
    # return each images as 96 x 96 x 1
    X = X.reshape(-1, 96,96, 1)

    y = df[df.columns[:-1]].values #(30 columns)
    # Normalize the target value, scale values between 0 and 1
    y = (y - 48) / 48
    # shuffle train data
    X, y = shuffle(X, y, random_state=42)
    y = y.astype(np.float32)

    return X,y
