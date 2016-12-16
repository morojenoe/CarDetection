import sys
import matplotlib.pyplot as plt
import numpy as np


def read_pgm(file_path):
    result = []
    with open(file_path, 'rb') as image:
        # skip header
        for i in range(14):
            image.read(1)

        for i in range(100 * 40):
            byte = image.read(1)
            result.append(int.from_bytes(byte, sys.byteorder))
    return result


def show_image(image):
    new_img = np.reshape(image, (40, 100))
    plt.imshow(new_img, cmap='gray')
    plt.show()


def get_data():
    path = 'CarData/TrainImages/'
    nb_neg = 500
    nb_pos = 550
    x_neg = [read_pgm(path + 'neg-' + str(i) + '.pgm') for i in range(nb_neg)]
    y_neg = [0]*nb_neg

    x_pos = [read_pgm(path + 'pos-' + str(i) + '.pgm') for i in range(nb_pos)]
    y_pos = [1] * nb_pos

    return x_neg + x_pos, y_neg + y_pos
