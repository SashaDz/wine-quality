import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.random import randn

# Basic constants in the lab
TOTAL_DATA = 1600
TRAIN = 1000
TEST = TOTAL_DATA - TRAIN
NEURONS = 30
EPOCHS = 100
learning_rate = 0.00001

# Read and split the data
data = pd.read_csv("winequality-red.csv", delimiter=';', header=0)
train_data = data.values[:TRAIN, :-1]
test_data = data.values[TRAIN:, :-1]
train_label = data.values[:TRAIN, -1]
test_label = data.values[TRAIN:, -1]

# Initialization weights and bias
weights1 = randn(NEURONS, 11)
weights2 = randn(1, NEURONS)
bias1 = randn(NEURONS, 1)
bias2 = randn(1, 1)

# Standardizing the data
for i in range(11):
    train_data[:][i] = (train_data[:][i] - train_data[:][i].mean()) / train_data[:][i].std()
    test_data[:][i] = (test_data[:][i] - test_data[:][i].mean()) / test_data[:][i].std()


# Sigmoid function
def sigmoid(x1):
    return 1 / (1 + np.exp(-x1))


# The function makes a predict
def predict(inputs, weights_1, weights_2, bias_1, bias_2):
    hidden_layer_ = np.array(sigmoid(np.dot(weights_1, inputs.T) + bias_1))
    outputs_ = np.dot(weights_2, hidden_layer_) + bias_2
    return outputs_


# The function returns the mean squared error
def forward_mean(inputs, weights_1, weights_2, bias_1, bias_2, label):
    return ((predict(inputs, weights_1, weights_2, bias_1, bias_2) - label) ** 2).mean()


# Arrays for plotting
time = []
loss_for_plot = []

# Perform training using gradient descent algorithm
for t in range(EPOCHS):
    hidden_layer = np.array(sigmoid(np.dot(weights1, train_data.T) + bias1))
    outputs = np.dot(weights2, hidden_layer) + bias2
    loss = np.square(outputs - train_label).sum()

    # Add elements to arrays for plotting
    time.append(t)
    loss_for_plot.append(loss)

    # Gradient computation with backpropagation
    grad_outputs = 2.0 * (outputs - train_label)
    grad_weights2 = np.dot(grad_outputs, hidden_layer.T)
    grad_bias2 = np.sum(grad_weights2, axis=1, keepdims=True)
    grad_hidden_layer = np.dot(weights2.T, grad_outputs)
    grad_weights1 = np.dot(grad_hidden_layer * hidden_layer * (1 - hidden_layer), train_data)
    grad_bias1 = np.sum(grad_weights1, axis=1, keepdims=True)

    # Gradient descent step
    weights1 -= learning_rate * grad_weights1
    weights2 -= learning_rate * grad_weights2
    bias1 -= learning_rate * grad_bias1
    bias2 -= learning_rate * grad_bias2

# Print accuracy and mean
match = 0
for i in range(TEST - 1):
    if test_label[i] == round(predict(test_data, weights1, weights2, bias1, bias2)[0][i]):
        match += 1
print('Accuracy = {}%'.format(round(match / (TEST - 1)*100, 1)))
print('Mean = {}'.format(round(forward_mean(test_data, weights1, weights2, bias1, bias2, test_label), 3)))

# Show plot time - loss
plt.plot(time[:50], loss_for_plot[:50])
plt.xlabel('time')
plt.ylabel('loss')
plt.show()
