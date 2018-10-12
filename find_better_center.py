from keras.datasets import mnist
import numpy as np
from matplotlib import pyplot as plt


def get_center(image):
    image = np.asarray(image)
    (rows, cols) = image.shape
    temp_image = np.copy(image)
    temp_image = np.where(temp_image > .5, temp_image, 0)

    total_mass = np.sum(np.sum(temp_image))
    col_sums = np.sum(temp_image, 0)
    x = np.sum(col_sums * np.arange(cols)) / total_mass
    return int(x)


def draw_center(image, center):
    image[:, center] = 255 * np.ones((image.shape[1]))
    return image


def main():
    (train_imgs, train_labels), (test_imgs, test_labels) = mnist.load_data()
    train_imgs = train_imgs.reshape(train_imgs.shape[0], 28, 28)
    for i in range(5):
        im = train_imgs[i]
        center = get_center(im)
        im = draw_center(im, center)
        plt.imshow(im)
        plt.gray()
        plt.show()


if __name__ == '__main__':
    main()
