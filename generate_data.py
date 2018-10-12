from image_stitcher import ImageStitcher
from data_iterator import Data_np
import numpy as np


def generate_data_np(train_length,
                     test_length,
                     validation_length,
                     train_images,
                     train_labels,
                     test_images,
                     test_labels,
                     batch_size,
                     singles_amount=.1,
                     testing=False,
                     singles=False,
                     doubles=True,
                     validation_split=0.2):

    image_split = len(train_images) - int(len(train_images) * validation_split)
    label_split = len(train_labels) - int(len(train_labels) * validation_split)

    val_images = train_images[image_split:]
    val_labels = train_labels[label_split:]

    train_images = train_images[:image_split]
    train_labels = train_labels[:label_split]

    test_images = test_images
    test_labels = test_labels

    val = ImageStitcher(
        56,
        val_images,
        val_labels,
        overlap_range=(-17, 0),
        repeated_digits=False,
        singles=singles,
        doubles=doubles,
        singles_amount=singles_amount,
        testing=testing)
    train = ImageStitcher(
        56,
        train_images,
        train_labels,
        overlap_range=(-17, 0),
        repeated_digits=False,
        singles=singles,
        doubles=doubles,
        singles_amount=singles_amount,
        testing=testing)
    test = ImageStitcher(
        56,
        test_images,
        test_labels,
        overlap_range=(-17, 0),
        repeated_digits=False,
        singles=singles,
        doubles=doubles,
        singles_amount=singles_amount,
        testing=testing)

    val.overlap_images(validation_length)
    train.overlap_images(train_length)
    test.overlap_images(test_length)

    return (Data_np((train.stitched_imgs, train.stitched_labels), batch_size),
            Data_np((val.stitched_imgs, val.stitched_labels), batch_size),
            Data_np((test.stitched_imgs, test.stitched_labels), batch_size))
