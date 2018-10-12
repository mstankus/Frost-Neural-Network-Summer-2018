import find_better_center as fc


def findOrder(image1, image2, label1, label2):
    center1 = fc.get_center(image1)
    center2 = fc.get_center(image2)
    if (center1 <= center2):
        return str(label1) + str(label2)
    else:
        return str(label2) + str(label1)


def main():
    (image1, labels1), (image2, label2) = mnist.load_data()
    stitcher = image_stitcher.ImageStitcher(
        40, image1, labels1, overlap_range=(-17, 0))
    for i in range(2):
        rand1 = random.randint(0, len(image1) - 1)
        rand2 = random.randint(0, len(image1) - 1)
        num_pixels = random.randint(-17, 0)
        img1 = stitcher.stitch(image1[rand1], np.zeros((28, 28), dtype=int),
                               num_pixels)
        img2 = stitcher.stitch(
            np.zeros((28, 28), dtype=int), image1[rand2], num_pixels)
        label1 = labels1[rand1]
        label2 = labels1[rand2]
        print('label1:' + str(label1))
        print('label2:' + str(label2))
        print(findOrder(img1, img2, label1, label2))


if __name__ == '__main__':
    import numpy as np
    import random
    from keras.datasets import mnist
    import image_stitcher
    main()
