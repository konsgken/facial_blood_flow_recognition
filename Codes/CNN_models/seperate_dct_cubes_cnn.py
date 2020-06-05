"""
Created on Fri Jan 24 19:42:58 2020

@author: kostasGk
"""
import warnings
import os

warnings.filterwarnings('ignore')

import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib

matplotlib.use("Agg")
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.models import Model
from scipy.io import loadmat, savemat
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Input
from keras.layers import concatenate
from keras.activations import relu
from keras.callbacks import ModelCheckpoint

# K.image_data_format() == "channels_last"
np.random.seed(0)  # Set a random seed for reproducibility


def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


def load(imagePath):
    data = []
    labels = []
    for (i, imagePath) in enumerate(imagePath):
        print(imagePath)
        label = imagePath.split(os.path.sep)[-2]
        print(label)
        mat_file = loadmat(imagePath)
        cube = mat_file['dct_output']  # cubes_together
        data.append(cube)
        labels.append(label)
    return (np.array(data), np.array(labels))


args_forehead = {
    'dataset': r'C:\Democritus University of Thrace (DUTh)\OneDrive\facial_blood_flow_recognition\Datasets\June2019\Three_Videos_Out\Amplified_Only_Blood_Flow_After_Attenuation'
               r'\DCT_cube_forehead\Day1and2',
    'testing': r'C:\Democritus University of Thrace (DUTh)\OneDrive\facial_blood_flow_recognition\Datasets\June2019\Three_Videos_Out\Amplified_Only_Blood_Flow_After_Attenuation'
               r'\DCT_cube_forehead\Day3'}

args_left_cheeck = {
    'dataset': r'C:\Democritus University of Thrace (DUTh)\OneDrive\facial_blood_flow_recognition\Datasets\June2019\Three_Videos_Out\Amplified_Only_Blood_Flow_After_Attenuation'
               r'\DCT_cube_left_cheeck\Day1and2',
    'testing': r'C:\Democritus University of Thrace (DUTh)\OneDrive\facial_blood_flow_recognition\Datasets\June2019\Three_Videos_Out\Amplified_Only_Blood_Flow_After_Attenuation'
               r'\DCT_cube_left_cheeck\Day3'}

args_right_cheeck = {
    'dataset': r'C:\Democritus University of Thrace (DUTh)\OneDrive\facial_blood_flow_recognition\Datasets\June2019\Three_Videos_Out\Amplified_Only_Blood_Flow_After_Attenuation'
               r'\DCT_cube_right_cheeck\Day1and2',
    'testing': r'C:\Democritus University of Thrace (DUTh)\OneDrive\facial_blood_flow_recognition\Datasets\June2019\Three_Videos_Out\Amplified_Only_Blood_Flow_After_Attenuation'
               r'\DCT_cube_right_cheeck\Day3'}

print("[INFO] loading cubes...")
imagePaths = list(getListOfFiles(args_forehead["dataset"]))
testimagePaths = list(getListOfFiles(args_forehead["testing"]))

(forehead_input_trainX, trainY) = load(imagePaths)
# forehead_input_trainX = forehead_input_trainX.astype("float")/255.0
(forehead_input_testX, testY) = load(testimagePaths)
# forehead_input_testX = forehead_input_testX.astype("float")/255.0

# forehead_input_trainX =forehead_input_trainX[0:64,:,:,:]
# forehead_input_testX =forehead_input_testX[0:64,:,:,:]

imagePaths = list(getListOfFiles(args_left_cheeck["dataset"]))
testimagePaths = list(getListOfFiles(args_left_cheeck["testing"]))

(left_cheeck_input_trainX, trainY) = load(imagePaths)
# left_cheeck_input_trainX = left_cheeck_input_trainX.astype("float")/255.0
(left_cheeck_input_testX, testY) = load(testimagePaths)
# left_cheeck_input_testX = left_cheeck_input_testX.astype("float")/255.0


imagePaths = list(getListOfFiles(args_right_cheeck["dataset"]))
testimagePaths = list(getListOfFiles(args_right_cheeck["testing"]))

(right_cheeck_input_trainX, trainY) = load(imagePaths)
# right_cheeck_input_trainX = right_cheeck_input_trainX.astype("float")/255.0
(right_cheeck_input_testX, testY) = load(testimagePaths)
# right_cheeck_input_testX = right_cheeck_input_testX.astype("float")/255.0


trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)
labelNames = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
# print(testY)
# %% Create model
chanDim = -1
classes = 10
# %%
forehead_input = Input(shape=(71, 202, 30), name='forehead_input')
left_cheeck_input = Input(shape=(71, 101, 30), name='left_cheeck_input')
right_cheeck_input = Input(shape=(71, 101, 30), name='right_cheeck_input')
# %%  CONV => RELU => CONV => RELU => POOL layer set
output_forehead_Conv2D_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(forehead_input)
output_left_cheeck_Conv2D_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(left_cheeck_input)
output_right_cheeck_Conv2D_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(right_cheeck_input)

output_forehead_BatchNormalization_1 = BatchNormalization(axis=chanDim)(output_forehead_Conv2D_1)
output_left_cheeck_BatchNormalization_1 = BatchNormalization(axis=chanDim)(output_left_cheeck_Conv2D_1)
output_right_cheeck_BatchNormalization_1 = BatchNormalization(axis=chanDim)(output_right_cheeck_Conv2D_1)

output_forehead_Conv2D_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(output_forehead_BatchNormalization_1)
output_left_cheeck_Conv2D_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(output_left_cheeck_BatchNormalization_1)
output_right_cheeck_Conv2D_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(output_right_cheeck_BatchNormalization_1)

output_forehead_BatchNormalization_2 = BatchNormalization(axis=chanDim)(output_forehead_Conv2D_2)
output_left_cheeck_BatchNormalization_2 = BatchNormalization(axis=chanDim)(output_left_cheeck_Conv2D_2)
output_right_cheeck_BatchNormalization_2 = BatchNormalization(axis=chanDim)(output_right_cheeck_Conv2D_2)

output_forehead_MaxPooling2D_1 = MaxPooling2D(pool_size=(2, 2))(output_forehead_BatchNormalization_2)
output_left_cheeck_MaxPooling2D_1 = MaxPooling2D(pool_size=(2, 2))(output_left_cheeck_BatchNormalization_2)
output_right_cheeck_MaxPooling2D_1 = MaxPooling2D(pool_size=(2, 2))(output_right_cheeck_BatchNormalization_2)

# output_forehead_Dropout_1=Dropout(0.25)(output_forehead_MaxPooling2D_1)
# output_left_cheeck_Dropout_1=Dropout(0.25)(output_left_cheeck_MaxPooling2D_1)
# output_right_cheeck_Dropout_1=Dropout(0.25)(output_right_cheeck_MaxPooling2D_1)
# CONV => RELU => CONV => RELU => POOL layer set
output_forehead_Conv2D_3 = Conv2D(64, (3, 3), padding='same', activation='relu')(output_forehead_MaxPooling2D_1)
output_left_cheeck_Conv2D_3 = Conv2D(64, (3, 3), padding='same', activation='relu')(output_left_cheeck_MaxPooling2D_1)
output_right_cheeck_Conv2D_3 = Conv2D(64, (3, 3), padding='same', activation='relu')(output_right_cheeck_MaxPooling2D_1)

output_forehead_BatchNormalization_3 = BatchNormalization(axis=chanDim)(output_forehead_Conv2D_3)
output_left_cheeck_BatchNormalization_3 = BatchNormalization(axis=chanDim)(output_left_cheeck_Conv2D_3)
output_right_cheeck_BatchNormalization_3 = BatchNormalization(axis=chanDim)(output_right_cheeck_Conv2D_3)

output_forehead_Conv2D_4 = Conv2D(64, (3, 3), padding='same', activation='relu')(output_forehead_BatchNormalization_3)
output_left_cheeck_Conv2D_4 = Conv2D(64, (3, 3), padding='same', activation='relu')(output_left_cheeck_BatchNormalization_3)
output_right_cheeck_Conv2D_4 = Conv2D(64, (3, 3), padding='same', activation='relu')(output_right_cheeck_BatchNormalization_3)

output_forehead_BatchNormalization_4 = BatchNormalization(axis=chanDim)(output_forehead_Conv2D_4)
output_left_cheeck_BatchNormalization_4 = BatchNormalization(axis=chanDim)(output_left_cheeck_Conv2D_4)
output_right_cheeck_BatchNormalization_4 = BatchNormalization(axis=chanDim)(output_right_cheeck_Conv2D_4)

output_forehead_MaxPooling2D_2 = MaxPooling2D(pool_size=(2, 2))(output_forehead_BatchNormalization_4)
output_left_cheeck_MaxPooling2D_2 = MaxPooling2D(pool_size=(2, 2))(output_left_cheeck_BatchNormalization_4)
output_right_cheeck_MaxPooling2D_2 = MaxPooling2D(pool_size=(2, 2))(output_right_cheeck_BatchNormalization_4)

# output_forehead_Dropout_2=Dropout(0.25)(output_forehead_MaxPooling2D_2)
# output_left_cheeck_Dropout_2=Dropout(0.25)(output_left_cheeck_MaxPooling2D_2)
# output_right_cheeck_Dropout_2=Dropout(0.25)(output_right_cheeck_MaxPooling2D_2)

output_forehead_Flatten = Flatten()(output_forehead_BatchNormalization_4)
output_left_cheeck_Flatten = Flatten()(output_left_cheeck_BatchNormalization_4)
output_right_cheeck_Flatten = Flatten()(output_right_cheeck_BatchNormalization_4)

concatenate_flatten = keras.layers.concatenate([output_forehead_Flatten, output_left_cheeck_Flatten, output_right_cheeck_Flatten])

output_Dense_1 = Dense(120, activation='relu')(concatenate_flatten)
output_BatchNormalization_flatten = BatchNormalization(axis=chanDim)(output_Dense_1)
# output_Dropout_flatten=Dropout(0.5)(output_BatchNormalization_flatten)
output = Dense(classes, activation='softmax')(output_BatchNormalization_flatten)

model = Model(inputs=[forehead_input, left_cheeck_input, right_cheeck_input], outputs=[output])

epochs = 20
print("[INFO] compiling model...")
opt = SGD(lr=0.01, decay=0.01 / epochs, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

print(model.summary())
filepath = "weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# train the network
print("[INFO] training network...")

H = model.fit([forehead_input_trainX, left_cheeck_input_trainX, right_cheeck_input_trainX], [trainY], epochs=epochs, batch_size=64,
              validation_data=([forehead_input_testX, left_cheeck_input_testX, right_cheeck_input_testX], testY), verbose=1, callbacks=callbacks_list)

# evaluate the network
print("[INFO] evaluating network...")
model.load_weights("weights.best.hdf5")
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])
predictions = model.predict([forehead_input_testX, left_cheeck_input_testX, right_cheeck_input_testX], batch_size=64)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=labelNames))
# print(model.summary())
# from keras.utils import plot_model
# plot_model(model, to_file='model.png')
