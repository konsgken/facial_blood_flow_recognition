# import the necessary packages
import cv2
from PIL import Image

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target image width, height, and interpolation
        # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # resize the image to a fixed size, ignoring the aspect
        # ratio

        # print("[SIMPLE PREPROCESSOR] image size: ", image.shape)

        # show training image as sample (just for code testing purposes)
        # img = Image.fromarray(image, 'RGB')
        # img.show()

        return cv2.resize(image, (self.width, self.height),
            interpolation=self.inter)