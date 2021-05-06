import gzip
import numpy as np


def get_digits_training_data(amount: int = 60000):
    label_data = gzip.open("trainingdata/train-labels-idx1-ubyte.gz").read()
    image_data = gzip.open("trainingdata/train-images-idx3-ubyte.gz").read()
    training_data = []
    for i in range(amount):
        print("{}/{} loaded".format(i, amount), end="\r")
        label = np.zeros(10)
        label[label_data[i + 8]] = 1
        image = np.empty(784)
        index = i * 784 + 16
        for j, k in enumerate(range(index, index + 784)):
            image[j] = image_data[k] / 255
        training_data.append((image, label))
    print("{} loaded".format(amount))
    return training_data


def get_digits_test_data(amount: int = 10000):
    label_data = gzip.open("testdata/t10k-labels-idx1-ubyte.gz").read()
    image_data = gzip.open("testdata/t10k-images-idx3-ubyte.gz").read()
    test_data = []
    for i in range(amount):
        print("{}/{} loaded".format(i, amount), end="\r")
        label = np.zeros(10)
        label[label_data[i + 8]] = 1
        image = np.empty(784)
        index = i * 784 + 16
        for j, k in enumerate(range(index, index + 784)):
            image[j] = image_data[k] / 255
        test_data.append((image, label))
    print("{} loaded".format(amount))
    return test_data
