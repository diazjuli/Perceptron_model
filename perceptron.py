import random
import math
import pandas as pd
from sommelier import dot_product


class Perceptron:
    def __init__(self):
        self.W = None
        self.learning_rate = 0
        self.performance = None

    def predict(self, xi):
        """
        This function returns the classfication for observation xi using our model's weights
        and a heavyside step activation function

        :param xi: vector
        :return: int: 1 or 0
        """

        result = dot_product(xi, self.W[1:]) + self.W[0]
        return 1 if result > 0.0 else 0

    def train_epoch(self, X, y, epoch):
        """
        This function trains one epoch of our algorithm.
        It uses a heavyside step activation function to classify each observation and
        the Rosenblatt Perceptron learning rule to update the weights.
        After training, we add the number of classification errors and the updated weights to
            the self.performance instance variable

        :param X: 2Dimensional array of observations: shape = (# of observations, # of attributes)
        :param y: pandas.series : the labels for our training set : shape = (# of observations)
        :param epoch: int : epoch #
        :return: int num_errors
        """

        num_errors = 0
        for xi, yi in zip(X, y):
            prediction = self.predict(xi)
            diff = yi - prediction
            self.W[0] += self.learning_rate * diff
            self.W[1:] += self.learning_rate * diff * xi
            num_errors += int(diff != 0.0)

        performance = (epoch, num_errors, self.W[1:], self.W[0])
        self.performance.append(performance)
        return num_errors

    def train(self, train_data, labels, num_epochs, learning_rate):
        """
        This function trains the perceptron num_epochs times, or until 0 errors if num_epochs is negative
        train_data has input values, labels are the output values, any learning rate can be specified

        :param train_data: matrix containing input variables
        :param labels: vector containing output variables as True or False values
        :param num_epochs: if num_epochs is 0, we train until there are 0 errors
        :param learning_rate:
        :return: array of tuples. Each tuple has the statistics for an epoch
                    (epoch, number of errors, [weight1, weight2], bias)
        """

        # initialize Perceptron instance variables, generate random numbers for weights and bias
        self.performance = []
        self.learning_rate = learning_rate
        self.W = [random.uniform(-1, 1) for i in range(train_data.shape[1] + 1)]

        epoch = 0
        if num_epochs < 0:
            raise ValueError("number of epochs must be non-negative")

        while True:
            errors = self.train_epoch(train_data, labels, epoch)
            if num_epochs == 0 and errors == 0:
                break
            if num_epochs != 0 and epoch >= num_epochs:
                break
            epoch += 1

        return self.performance
