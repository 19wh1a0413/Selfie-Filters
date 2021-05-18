import numpy as np
import pandas as pd
import cv2 , os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def create_cnn_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (5, 5), activation = 'relu', input_shape = (96, 96, 1)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(256, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))

    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation = 'relu'))
    model.add(layers.Dense(256, activation = 'relu'))
    model.add(layers.Dense(128, activation = 'relu'))
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(32, activation = 'relu'))
    return model

def compile_cnn_model(model, optimizer, loss, metrics):
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

def train_cnn_model(model, X_train, y_train):
    return model.fit(X_train, y_train) #epochs = 100, batch_size = 200, verbose = 1, validation_split = 0.2)

def load_cnn_model(fileName):
    return models.load_model(fileName)

def save_cnn_model(model, fileName):
    model.save(fileName + '.h5')

def train_test_split(df):
    df['Image'] = df['Image'].apply(lambda x: np.fromstring(x, sep=' '))
    df = df.dropna()
    X = np.vstack(df['Image'].values) / 255
    X = X.astype(np.float32)
    X = X.reshape(-1, 96, 96, 1)
    y = df[df.columns[:-1]].values
    y = (y - 48) / 48
    X, y = shuffle(X, y, random_state = 40)
    y = y.astype(np.float32)
    return X, y
