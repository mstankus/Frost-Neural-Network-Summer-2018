#!/usr/bin/python
from __future__ import division
import matplotlib
matplotlib.use('agg')

import tensorflow as tf
from data_iterator import Data_np
from generate_data import generate_data_np
import numpy as np
np.set_printoptions(threshold=1568)

from keras import models
from keras.datasets import mnist

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import order_finder as of


# used for repeated digits
def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=keep_dims)
    return tf.sqrt(squared_norm + epsilon)


def masker(masker, digit_caps):
    reshaped_masker = tf.reshape(masker, [-1, 1, 10, 1, 1])
    masked_capsule = tf.multiply(reshaped_masker, digit_caps)
    final_masked = tf.reshape(masked_capsule, [-1, 160])
    return final_masked


def main():
    # Represents the number of imgs used in testing (this is now just used for saving images)
    training_size = 60000
    # Represents the input size used during testing
    input_size = 100

    (train_images, train_labels), (test_images,
                                   test_labels) = mnist.load_data()
    train_data, val_data, test_data = generate_data_np(
        input_size,
        input_size,
        input_size,
        train_images,
        train_labels,
        test_images,
        test_labels,
        input_size,
        singles=True,
        doubles=True,
        testing=True)

    imgs = test_data.data[0]
    labels = test_data.data[1]

    # load 1 vs. 2 digit classifier (cams_net)
    loaded_model = models.load_model('model.h5')
    singles_or_doubles = loaded_model.predict(imgs)

    singles_or_doubles = np.where(singles_or_doubles > .5, 1, 0)

    single_imgs = np.delete(imgs, np.nonzero(singles_or_doubles), axis=0)
    single_labels = np.delete(labels, np.nonzero(singles_or_doubles), axis=0)

    singles_or_doubles = np.where(singles_or_doubles == 1, 0, 1)

    double_imgs = np.delete(imgs, np.nonzero(singles_or_doubles), axis=0)
    double_labels = np.delete(labels, np.nonzero(singles_or_doubles), axis=0)

    with tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)) as sess:
        signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        model_dir = "/home/chfredri/DoubleDigit/double_caps_1gpu"
        meta_graph = tf.saved_model.loader.load(sess, ["Post Training"],
                                                model_dir)

        input_key = 'X'
        label_key = 'y'
        output_pred_key = 'y_pred'
        decoder_input_key = 'decoder_input'
        decoder_output_key = 'decoder_output'
        casp2_output_key = 'caps2_output'

        signature = meta_graph.signature_def
        x_tensor_name = signature[signature_key].inputs[input_key].name
        y_tensor_name = signature[signature_key].inputs[label_key].name
        y_pred_tensor_name = signature[signature_key].outputs[
            output_pred_key].name
        decoder_input_tensor_name = signature[signature_key].inputs[
            decoder_input_key].name
        decoder_output_tensor_name = signature[signature_key].outputs[
            decoder_output_key].name
        caps2_output_tensor_name = signature[signature_key].outputs[
            casp2_output_key].name

        x = sess.graph.get_tensor_by_name(x_tensor_name)
        y = sess.graph.get_tensor_by_name(y_tensor_name)
        y_pred = sess.graph.get_tensor_by_name(y_pred_tensor_name)
        decoder_input = sess.graph.get_tensor_by_name(
            decoder_input_tensor_name)
        decoder_output = sess.graph.get_tensor_by_name(
            decoder_output_tensor_name)
        caps2_output = sess.graph.get_tensor_by_name(caps2_output_tensor_name)

        X_batch, y_batch = (double_imgs, double_labels)
        feed_dict = {x: X_batch.reshape([-1, 28, 56, 1])}
        digit_caps, masking_indices = sess.run(
            fetches=[caps2_output, y_pred], feed_dict=feed_dict)

        y_prob = safe_norm(digit_caps, axis=-2, name="y_prob")
        y_prob_squeezed = tf.squeeze(y_prob, axis=-1)
        top_2_capsules = tf.nn.top_k(y_prob_squeezed, 2, sorted=True)
        top_2_capsules = tf.squeeze(top_2_capsules.values)

        reconstruction_mask = tf.one_hot(masking_indices, depth=10)
        first_masker = reconstruction_mask[:, 0]
        second_masker = reconstruction_mask[:, 1]

        first_set = masker(first_masker, digit_caps)
        first = np.array(sess.run(first_set))

        second_set = masker(second_masker, digit_caps)
        second = np.array(sess.run(second_set))

        feed_dict_1 = {decoder_input: first}
        feed_dict_2 = {decoder_input: second}

        output_1 = sess.run(fetches=decoder_output, feed_dict=feed_dict_1)
        output_2 = sess.run(fetches=decoder_output, feed_dict=feed_dict_2)

        first_pictures = output_1.reshape([-1, 28, 56])
        second_pictures = output_2.reshape([-1, 28, 56])

        count = 0
        for i in range(double_labels.shape[0]):
            if None in double_labels[i]:
                label = str(double_labels[i][0])
            else:
                label = str(double_labels[i][0]) + str(double_labels[i][1])

            image_one = np.array(first_pictures[i])
            image_two = np.array(second_pictures[i])

            image_one = np.reshape(image_one, [28, 56])
            image_one[image_one > 255] = 255
            image_two = np.reshape(image_two, [28, 56])
            image_two[image_two > 255] = 255

            # order_finder test
            ans = label

            if None in double_labels[i]:
                pred = masking_indices[i, 0]
            else:
                if top_2_capsules[i].eval()[1] < .5:  # repeated digit threshold
                    pred = str(masking_indices[i, 0]) + str(
                        masking_indices[i, 0])
                else:
                    pred = of.findOrder(image_one, image_two,
                                        masking_indices[i, 0],
                                        masking_indices[i, 1])
            if ans == pred:
                count += 1
            else:
                print('ans', ans, 'pred', pred)
            # file_name = str(training_size) + str(i) + "prediction_" + str(
            #     pred) + "_label_" + str(label)
            # plt.matshow(image_one, aspect='auto', cmap='gray')
            # plt.savefig(file_name)
            # plt.close()
            # plt.savefig(multi_images,format='pdf')
            '''plt.matshow(image_two, aspect='auto', cmap='gray')
    		plt.savefig(multi_images,format='pdf')'''
        double_acc = count / double_imgs.shape[0]
        print("double_acc:", double_acc)

        # ################### Handles the single digit list###################################
        # # Setting up meta data
        signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        model_dir = "/home/chfredri/DoubleDigit/single_caps_1gpu"
        meta_graph_single = tf.saved_model.loader.load(sess, ["Post Training"],
                                                       model_dir)

        input_key = 'X'
        label_key = 'y'
        output_pred_key = 'y_pred'

        signature_single = meta_graph_single.signature_def
        single_X_tensor_name = signature_single[signature_key].inputs[
            input_key].name
        single_y_tensor_name = signature_single[signature_key].inputs[
            label_key].name
        single_y_pred_tensor_name = signature_single[signature_key].outputs[
            output_pred_key].name

        single_X = sess.graph.get_tensor_by_name(single_X_tensor_name)
        single_y = sess.graph.get_tensor_by_name(single_y_tensor_name)
        single_y_pred = sess.graph.get_tensor_by_name(
            single_y_pred_tensor_name)

        # Making predictions on new data
        feed_dict = {single_X: np.reshape(single_imgs, [-1, 28, 56, 1])}
        predictions = sess.run(fetches=single_y_pred, feed_dict=feed_dict)

        # Sums up the times when predictions and single_labels are the same
        single_acc = np.sum(predictions == single_labels) / single_imgs.shape[0]
        print("single_acc:", single_acc)
        print(single_imgs.shape[0])

        final_accuracy = ((single_acc * single_imgs.shape[0]) +
                          (double_acc * double_imgs.shape[0])) / input_size
        print("final accuracy:", final_accuracy)


if __name__ == '__main__':
    main()
    print("WE ARE DONE!")
