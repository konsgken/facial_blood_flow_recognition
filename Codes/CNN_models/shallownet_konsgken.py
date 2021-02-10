# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report
from Codes.CNN_models.preprocessing.simplepreprocessor import SimplePreprocessor
from Codes.CNN_models.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from Codes.CNN_models.datasets.simpledatasetloader import SimpleDatasetLoader
from Codes.CNN_models.nn.conv.shallownet import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import numpy as np

np.random.seed(2020)  # Set a random seed for reproducibility
args = {
    'dataset': r'C:\Democritus University of Thrace (DUTh)\OneDrive\facial_blood_flow_recognition\Datasets\June2019\Three_Videos_Out\Amplified_Only_Blood_Flow_After_Attenuation'
               r'\Together_Avg_Images_Gray\Day1and2',
    'validation': r'C:\Democritus University of Thrace (DUTh)\OneDrive\facial_blood_flow_recognition\Datasets\June2019\Three_Videos_Out'
                  r'\Amplified_Only_Blood_Flow_After_Attenuation\Together_Avg_Images_Gray\Day3',
    }
# grab the list of images that we'll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
valimagePaths = list(paths.list_images(args["validation"]))

# initialize the image preprocessor
sp = SimplePreprocessor(202, 142)
iap = ImageToArrayPreprocessor()

# load the dataset from disk the scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(trainX, trainY) = sdl.load(imagePaths, verbose=500)
trainX = trainX.astype("float")/255.0

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
#(trainX, valX, trainY, valY) = train_test_split(trainX, trainY,
#             test_size=0.25, random_state=42)

print("[INFO] size of training data: ", trainX.shape)
#print("[INFO] size of validation data: ", valX.shape)

(testX, testY) = sdl.load(valimagePaths, verbose=500)
testX = testX.astype("float")/255.0

print("[INFO] size of testing data: ", testX.shape)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
#valY = LabelBinarizer().fit_transform(valY)
testY = LabelBinarizer().fit_transform(testY)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.005)
model = ShallowNet.build(width=202, height=142, depth=1, classes=5)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY,
              batch_size=32, epochs=10, verbose=1)
#validation_data=(valX, valY)
# save the network to disk (HDF5 format)
print("[INFO] serializing network...")

# # load the pre-trained network
# print("[INFO] loading pre-trained network...")
# model = load_model(args["model"])

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
        predictions.argmax(axis=1),
        target_names=["01", "02", "03", "04", "05"]))

# plot the training loss and accuracy
#plt.style.use("ggplot")
#plt.figure()
#plt.plot(np.arange(0, 50), H.history["loss"], label="train_loss")
#plt.plot(np.arange(0, 50), H.history["val_loss"], label="val_loss")
#plt.plot(np.arange(0, 50), H.history["acc"], label="train_acc")
#plt.plot(np.arange(0, 50), H.history["val_acc"], label="val_acc")
#plt.title("Training Loss and Accuracy")
#plt.xlabel("Epoch #")
#plt.ylabel("Loss/Accuracy")
#plt.legend()
#plt.show()