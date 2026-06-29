"""
First Neural Network

A simple feedforward neural network built using TensorFlow and Keras
to classify handwritten digits from the MNIST dataset.

Author: Vasant Desai
GitHub: https://github.com/vasantdesai212-dotcom

Test Accuracy: 97.58%
"""
import tensorflow as tf
from tensorflow import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data() #Here, x-train means the photos or the images on which the data is being trained, and y is the exact output or the answer of that image. In the same manner, there is x-test and y-test for the testing data. 
x_train = x_train / 255.0  #Here we minimize the numeric value of the training data by dividing it by 255, just to make sure that the code is handled in a efficient way. 
x_test = x_test / 255.0
import matplotlib.pyplot as plt 
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), #The image is a matrix of 28 by 28, but the neural network expects a flat vector. This function, the 'dot flatten' function, converts the 28 by 28 matrix into a 784 vector. 
    keras.layers.Dense(128, activation='relu'), #The dense(128) means that we are creating 128 neurons for 784 inputs. The number could be anything, but let's say it's 128. The activation equals ReLU, which means we are activating the ReLU function so that the neural network is not just a static linear equation that stops but actually repeats and learns. 
    keras.layers.Dense(10, activation='softmax') #The dense(10) means that we are creating 10 output neurons that actually give the probability of the prediction of the number. The probability here has been predicted and calculated using the activation = softmax function. The softmax actually turns the larger numbers into smaller probabilities that are between 0 and 1. 
])

model.compile( #The neural network asks three questions. How should I learn? How should I measure my mistakes? How should I evaluate myself? The compile() helps answer all these. 
    optimizer='adam', # Here, the optimizer is the function that actually adjusts the weights and biases by the method of back propagation, and Adam is one of the most popular optimizers used for beginners, specially. 
    loss='sparse_categorical_crossentropy', #This is a loss function that actually tells the neural network how wrong it was. We use sparse here because our outputs are stored as integers. Otherwise, we would have only used categorical cross-entropy. 
    metrics=['accuracy'] #This tells the model to show the accuracy of the predictions. 
)

model.fit(x_train, y_train, epochs=5) #Here, the .fit function actually tell the model how to approach the learning process. The x-train is the image that the model will look at and then compare it with y-train. This repeats five times, and that's why we have put epochs=5. 

test_loss, test_accuracy = model.evaluate(x_test, y_test) # Here, the model evaluates the X test data and checks with the Y test data and actually returns the loss and accuracy. Loss is an inbuilt feature of evaluate, but accuracy we have defined in the compile of the model. 
print("test accuracy:", test_accuracy)

predictions = model.predict(x_test) #Here, the model predicts what the output is using the X-test data. 

import numpy as np #The predictions would be in the array of 10 probabilities for a single number, but the answer is the one with maximum probability, and that's why we are using numpy here. 
predicted_num = np.argmax(predictions[0]) #The argmax function picks up the probability that is the maximum probability and gives the index of that. That would be the answer or the predicted digit. 
print("predicted digit: ", predicted_num)
print("actual digit: ", y_test[0])

plt.imshow(x_test[0], cmap='grey') #Here, the imshows and is helpful to preview the image. The image of X-train on index zero, and the cmap means the colour map, and that is grayscale. 
plt.show()
