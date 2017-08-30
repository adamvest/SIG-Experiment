'''
ML/DL SIG experiment 1
8/29
KNN for flower classification
'''

import numpy as np
import random

#change this
data_path = "/casa/adam/Downloads/iris_data.txt"
classname_to_int = {"Iris-setosa": 0, "Iris-versicolor":1, "Iris-virginica":2}
k = 5
train_test_split_ratio = .7

def load_data():
    f = open(data_path, "r")
    lines = f.readlines()
    f.close()

    x, y = [], []

    for line in lines:
        if line != "":
            data = line.strip().split(",")

            if data[-1] != "":
                data, class_num = data[:-1], data[-1]
                x.append([float(val) for val in data])
                y.append(classname_to_int[class_num])

    data = zip(x, y)
    random.shuffle(data)
    split_index = int(train_test_split_ratio * len(data))
    train, test = data[:split_index], data[split_index:]
    x_train, y_train = [np.array(num[0]) for num in train], [num[1] for num in train]
    x_test, y_test = [np.array(num[0]) for num in test], [num[1] for num in test]

    return x_train, y_train, x_test, y_test

def test_point(x_train, y_train, test_features):
    min_dists, class_vals, scores = [], [], []

    for i in range(len(x_train)):
        dist = np.linalg.norm(x_train[i] - test_features)

        if len(min_dists) < k or dist < min_dists[0][0]:
            if len(min_dists) >= k:
                min_dists.pop(0)

            min_dists.append((dist, i))

        min_dists.sort(reverse=True)

    for pt in min_dists:
        class_vals.append(y_train[pt[1]])

    for i in range(len(classname_to_int)):
        scores.append(class_vals.count(i))

    return scores.index(max(scores))

def test(x_train, y_train, x_test, y_test):
    correct_total = 0

    for i in range(len(x_test)):
        pred = test_point(x_train, y_train, x_test[i])
        correct_total += 1 if (pred == y_test[i]) else 0

    print "Accuracy of KNN Classifier: " + str(float(correct_total) / len(x_test))

x_train, y_train, x_test, y_test = load_data()
test(x_train, y_train, x_test, y_test)
