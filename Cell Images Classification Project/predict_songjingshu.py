import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import cohen_kappa_score, f1_score
from keras.initializers import glorot_uniform
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.datasets import mnist
from PIL import Image
import glob
from keras.models import load_model



def predict(x):
    # Here x is a NumPy array. On the actual exam it will be a list of paths.
    # %% --------------------------------------------- Data Prep -------------------------------------------------------
    imgs = []
    count = 0
    mean = np.zeros(3, np.int64)
    for file in x:
        if file.endswith(".png"):
            img = cv2.imread(file)
            count += 1
            mean += np.sum(img, axis=(0, 1)).astype(int)
            h, w = img.shape[0:-1]
            means = mean / (1.0 * h * w * count)
            img_1 = img - means
            #b = sum(img)
            #mean_image = np.mean(b)
            #img = b - mean_image
            resized_img = cv2.resize(img_1, (90, 90))
            img_resize = np.array(resized_img)
            imgs.append(img_resize.flatten())

    x = np.array(imgs)
    x = x / 255
    # Write any data prep you used during training
    # %% --------------------------------------------- Predict ---------------------------------------------------------
    model = load_model('mlp_songjingshu.hdf5')
    # If using more than one model to get y_pred, they need to be named as "mlp_ajafari1.hdf5", ""mlp_ajafari2.hdf5", etc.
    y_pred = np.argmax(model.predict(x), axis=1)
    return y_pred, model

    # If using more than one model to get y_pred, do the following:
    # return y_pred, model1, model2  # If you used two models
    # return y_pred, model1, model2, model3  # If you used three models, etc.
x_test = ["train/cell_1.png", "train/cell_2.png", "train/cell_10002.png", "train/cell_12288.png"]  # Dummy image path list placeholder
y_test_pred, *models = predict(x_test)

    # %% -------------------------------------------------------------------------------------------------------------------
assert isinstance(y_test_pred, type(np.array([1])))  # Checks if your returned y_test_pred is a NumPy array
assert y_test_pred.shape == (len(x_test),)  # Checks if its shape is this one (one label per image path)
# Checks whether the range of your predicted labels is correct
assert np.unique(y_test_pred).max() <= 3 and np.unique(y_test_pred).min() >= 0

print(y_test_pred)





