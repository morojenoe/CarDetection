import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split


def read_pgm(file_path):
    result = []
    with open(file_path, 'rb') as image:
        # skip header
        for i in range(14):
            image.read(1)

        for i in range(100 * 40):
            byte = image.read(1)
            result.append(int.from_bytes(byte, sys.byteorder))
    return np.reshape(result, (40, 100, 1))


def show_image(image):
    new_img = np.reshape(image, (40, 100))
    plt.imshow(new_img, cmap='gray')
    plt.show()


def get_data():
    path = 'CarData/TrainImages/'
    nb_neg = 500
    nb_pos = 550
    x_neg = [read_pgm(path + 'neg-' + str(i) + '.pgm') for i in range(
        nb_neg)]
    y_neg = [[1, 0]] * nb_neg

    x_train_neg, x_test_neg, y_train_neg, y_test_neg = train_test_split(
        x_neg, y_neg, test_size=0.3, random_state=42)

    x_pos = [read_pgm(path + 'pos-' + str(i) + '.pgm') for i in range(nb_pos)]
    y_pos = [[0, 1]] * nb_pos
    x_train_pos, x_test_pos, y_train_pos, y_test_pos = train_test_split(
        x_pos, y_pos, test_size=0.3, random_state=42)

    return x_train_neg + x_train_pos, y_train_neg + y_train_pos, \
           x_test_neg + x_test_pos, y_test_neg + y_test_pos
