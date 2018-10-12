#!/usr/bin/python
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import random
import time
from image_stitcher import ImageStitcher
from data_iterator import Data_np
from generate_data import generate_data_np
from keras import models


def create_2digit_1digit_data(imgs_1d, imgs_2d, num_imgs):
    new_x, new_y = np.zeros((num_imgs, 28, 56)), np.zeros(num_imgs)
    for img in range(num_imgs):
        idx = random.randint(0, num_imgs - 1)
        if random.random() < 0.5:
            new_x[img] = imgs_1d[idx]
            new_y[img] = 0
        else:
            new_x[img] = imgs_2d[idx]
            new_y[img] = 1
    return new_x, new_y


def main():
    # Extracting data from mnist
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images,
                                   test_labels) = mnist.load_data()
    batch_size = 32
    train_data, val_data, test_data = generate_data_np(
        60000,  # training size
        10000,  # testing size
        5000,   # validation size
        train_images,
        train_labels,
        test_images,
        test_labels,
        batch_size,
        singles=True,
        doubles=True,
        testing=True)

    train_images, train_labels = train_data.data[0], train_data.data[1]
    test_images, test_labels = test_data.data[0], test_data.data[1]

    new_train_labels = []
    # labels are 0 for single digits and 1 for double digits
    for label in train_labels:
        if None in label:
            new_train_labels.append(0)
        else:
            new_train_labels.append(1)
    train_labels = np.array(new_train_labels)

    new_test_labels = []
    for label in test_labels:
        if None in label:
            new_test_labels.append(0)
        else:
            new_test_labels.append(1)
    test_labels = np.array(new_test_labels)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(
            28,
            56,
        )),
        tf.keras.layers.Dense(8, activation=tf.nn.relu),
        tf.keras.layers.Dense(16, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=2, batch_size=batch_size)
    results = model.evaluate(test_images, test_labels)
    print(model.metrics_names, results)

    model.save("//home/chfredri/Summer-Research-2018/model.h5")
    print("Saved model to disk")


if __name__ == '__main__':
    beginTime = time.time()
    main()
    endTime = time.time()
    print('Total time: {:5.2f}s'.format(endTime - beginTime))
