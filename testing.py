from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import tensorflow.keras as keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import tensorflow as tf
from keras.utils import np_utils
from keras.models import load_model
from keras.datasets import cifar10
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#import tensorflow as tf



(_ ,_ ), (x_test, y_test) = cifar10.load_data()

x_test = x_test / 255.0
y_test = np_utils.to_categorical(y_test, 10)

model = load_model('best_model.hdf5')
model.load_weights('best_model.hdf5')




results = model.evaluate(x_test, y_test)

print("Loss of test model is  ", results[0])
print("Accuracy of test model is ", results[1]*100, "%")
