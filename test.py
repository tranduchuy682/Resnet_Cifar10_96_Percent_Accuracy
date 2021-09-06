
from keras.utils import np_utils
from keras.models import load_model
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

(_ ,_ ), (x_test, y_test) = cifar10.load_data()

x_test = x_test / 255.0
y_test = np_utils.to_categorical(y_test, 10)

model = load_model('best_model.hdf5')
model.load_weights('best_model.hdf5')

results = model.evaluate(x_test, y_test)

print("Loss of test model is  ", results[0])
print("Accuracy of test model is ", results[1]*100, "%")
