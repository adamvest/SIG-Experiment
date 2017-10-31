'''
ML/DL SIG experiment 4
10/21
Linear Regression using Gradient Descent
'''


import argparse
import numpy as np
from sklearn import datasets
from pandas import DataFrame


class LinearRegressionOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--num_epochs', type=int, default=1000, help="number of epochs to train for")
        self.parser.add_argument('--batch_size', type=int, default=16, help="number of samples per batch")
        self.parser.add_argument('--lr', type=float, default=.001, help="gradient descent step size")
        self.parser.add_argument('--reg_strength', type=float, default=1e-3, help="L2 regularization strength")
        self.parser.add_argument('--train_test_split_ratio', type=float, default=.8, help="percentage of data to use for training")

    def parse_args(self):
        return self.parser.parse_args()


#train a linear regression model
def train_linear_regression_model(data_x, data_y, lr, num_epochs, batch_size, reg_strength):
    weights = np.random.normal(0, .1, len(data_x[0]))
    num_batches = len(data_x) / batch_size

    for i in range(1, num_epochs+1):
        print "\nBegin Training Epoch %d" % i

        for j in range(num_batches):
            batch_x = data_x[j * batch_size:(j+1) * batch_size]
            batch_y = data_y[j * batch_size:(j+1) * batch_size]
            batch_preds = np.dot(weights, batch_x.T)
            diffs = batch_preds - batch_y
            batch_loss = np.sum(.5 * np.square((diffs))) + reg_strength * np.sum(np.square(weights))
            grad = sum([diffs[i] * batch_x[i] for i in range(batch_size)]) + 2 * reg_strength * weights
            weights -= lr * grad

            print "Batch %d Loss: %g" % (j+1, batch_loss)

    return weights


if __name__ == "__main__":
    args = LinearRegressionOptions().parse_args()

    diabetes = datasets.load_diabetes()
    data_x, data_y = diabetes.data, diabetes.target
    data = np.column_stack((np.ones((len(data_x), 1)), data_x, data_y))
    np.random.shuffle(data)
    idx = int(args.train_test_split_ratio*len(data))
    train, test = data[:idx], data[idx:]
    train_x, train_y = train[:, :-1], train[:, -1]
    test_x,  test_y = test[:, :-1], test[:, -1]

    optimal_weights = train_linear_regression_model(train_x, train_y, args.lr, args.num_epochs, args.batch_size, args.reg_strength)

    for i in range(len(test_x)):
        print "\nPredicted Value:", np.dot(optimal_weights, test_x[i].T)
        print "Actual Value:", test_y[i]
        raw_input('stop')
