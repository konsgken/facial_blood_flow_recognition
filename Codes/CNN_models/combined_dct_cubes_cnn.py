# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:32:24 2019

@author: konsg
"""

# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from nn.conv.minivggnet import MiniVGGNet
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import load_model
from keras.models import Model


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
        cube = mat_file['cubes_together']
        data.append(cube)
        labels.append(label)
    return np.array(data), np.array(labels)


args = {
    'dataset': r'C:\Democritus University of Thrace (DUTh)\OneDrive\facial_blood_flow_recognition\Datasets\June2019\Three_Videos_Out\Amplified_Only_Blood_Flow_After_Attenuation'
               r'\DCT_cube_together\Day1and2',
    'testing': r'C:\Democritus University of Thrace (DUTh)\OneDrive\facial_blood_flow_recognition\Datasets\June2019\Three_Videos_Out\Amplified_Only_Blood_Flow_After_Attenuation'
               r'\DCT_cube_together\Day3'}

print("[INFO] loading cubes...")
# imagePaths = list(paths.list_images(args["dataset"]))
imagePaths = list(getListOfFiles(args['dataset']))
testimagePaths = list(getListOfFiles(args['testing']))
(trainX, trainY) = load(imagePaths)
print("[INFO] size of training data: ", trainX.shape)
# print("[INFO] size of validation data: ", valX.shape)
(testX, testY) = load(testimagePaths)
print("[INFO] size of testing data: ", testX.shape)
# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
# valY = LabelBinarizer().fit_transform(valY)
testY = LabelBinarizer().fit_transform(testY)
labelNames = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

# initialize the optimizer and model
epochs = 10
print("[INFO] compiling model...")
opt = SGD(lr=0.01, decay=0.01 / epochs, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=202, height=142, depth=30, classes=10)  # dct together width=202, height=142, depth=29, classes=5
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=64, epochs=epochs, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=labelNames))
print(model.summary())

# plot the training loss and accuracy
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, epochs), H.history["acc"], label="train_acc")
# plt.plot(np.arange(0, epochs), H.history["val_acc"], label="val_acc")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend()
