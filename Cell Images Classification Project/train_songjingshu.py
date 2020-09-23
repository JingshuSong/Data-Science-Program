import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import cohen_kappa_score, f1_score
from keras.initializers import glorot_uniform
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.datasets import mnist
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import cross_validation
imgs = []
labels = []
count = 0
mean = np.zeros(3, np.int64)
for file in os.scandir("train"):
    if file.name.endswith(".png"):
        img = cv2.imread(os.path.join("train", file.name))
        count += 1
        mean += np.sum(img, axis=(0, 1)).astype(int)
        h, w = img.shape[0:-1]
        means = mean / (1.0 * h * w * count)
        img_1 = img - means
        resized_img = cv2.resize(img_1,(90,90))
        imgs.append(list(resized_img.flatten()))
        txt = file.name[:-4]+'.txt'
        a = open(os.path.join("train", txt))
        label = a.read()
        labels.append(label)
        a.close()
x, y = np.array(imgs), np.array(labels)
le = LabelEncoder()
le.fit(["red blood cell", "ring", "schizont", "trophozoite"])
y = le.transform(y)
y = to_categorical(y, num_classes = 4)

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_uniform(seed=SEED)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_NEURONS = (100, 200, 100, 150)
N_EPOCHS = 40
BATCH_SIZE = 1500
DROPOUT = 0.1
# %% -------------------------------------- Data Prep ------------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=SEED, test_size=0.2, stratify=y)
x_train, x_test = x_train.reshape(len(x_train), -1), x_test.reshape(len(x_test), -1)
x_train, x_test = x_train/255, x_test/255
ros = RandomOverSampler(random_state=0)
x_train,y_train = ros.fit_sample(x_train,y_train)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = Sequential([  # The dropout is placed right after the outputs of the hidden layers.
    Dense(N_NEURONS[0], input_dim=x.shape[1], kernel_initializer=weight_init),  # This sets some of these outputs to 0, so that
    Activation("relu"),  # a random dropout % of the hidden neurons are not used during each training step,
    Dropout(DROPOUT),  # nor are they updated. The Batch Normalization normalizes the outputs from the hidden
    BatchNormalization()  # activation functions. This helps with neuron imbalance and can speed training significantly.
])  # Note this is an actual layer with some learnable parameters. It's not just min-maxing or standardizing.
# Loops over the hidden dims to add more layers

model.add(Dense(N_NEURONS[1], activation="relu", kernel_initializer=weight_init))
model.add(Dropout(DROPOUT, seed=SEED))
model.add(BatchNormalization())


model.add(Dense(N_NEURONS[2], activation="softmax", kernel_initializer=weight_init))
model.add(Dropout(DROPOUT, seed=SEED))
model.add(BatchNormalization())

model.add(Dense(N_NEURONS[3], activation="tanh", kernel_initializer=weight_init))
model.add(Dropout(DROPOUT, seed=SEED))
model.add(BatchNormalization())

# Adds a final output layer with softmax to map to the 4 classes
model.add(Dense(4, activation="softmax", kernel_initializer=weight_init))
model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])
#es = EarlyStopping(monitor='val_loss',  patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
#mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
# %% -------------------------------------- Training Loop ----------------------------------------------------------
# Trains the MLP, while printing validation loss and metrics at each epoch
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test),
          callbacks=[ModelCheckpoint("mlp_songjingshu.hdf5", monitor="val_loss", save_best_only=True)])

# %% ------------------------------------------ Final test -------------------------------------------------------------
print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")
print("Cohen Kappa", cohen_kappa_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1)))
print("F1 score", f1_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1), average = 'macro'))



