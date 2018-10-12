#!/usr/bin/python
from __future__ import division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import os
from keras.datasets import mnist
from data_iterator import Data_np
from generate_data import generate_data_np

BATCH_SIZE_TRAIN = 128
TRAIN_LENGTH = 2048
VALIDATION_LENGTH = 256
TEST_LENGTH = 512
NUM_EPOCHS = 2
NUM_GPUS = 2


def format_time(seconds):
    # fast implementation of getting hrs:min:sec
    secs = int(seconds)
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    return "{:02d}:{:02d}:{:02d}".format(h, m, s)


def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector


# Computes Length of vectors in Capsule
def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(
            tf.square(s), axis=axis, keepdims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)


# k represents how many digits we are looking for
# The first capsule layer will have 32 * 6 * 6 capsules of 8 dimensions each
def make_model(num_gpus,
               device_index,
               X,
               y,
               k=2,
               caps1_n_maps=32,
               caps1_n_dims=8):
    parallel_batch_size = tf.shape(X)[0]
    caps1_n_caps = caps1_n_maps * 6 * 20
    with tf.device(
            tf.DeviceSpec(device_type="GPU", device_index=device_index)):
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            # -----First two Convolutions-----
            conv1_params = {
                "filters": 256,
                "kernel_size": 9,
                "strides": 1,
                "padding": "valid",
                "activation": tf.nn.relu,
            }

            conv2_params = {
                "filters": caps1_n_maps * caps1_n_dims,
                "kernel_size": 9,
                "strides": 2,
                "padding": "valid",
                "activation": tf.nn.relu,
            }

            conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
            conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)

            # -----Forming data into Primary Capsule Layer-----
            # Note the epsilon arg. We add it to squared_norm so that we never divide by zero
            caps1_raw = tf.reshape(
                conv2, [-1, caps1_n_caps, caps1_n_dims], name="caps1_raw")

            caps1_output = squash(caps1_raw, name="caps1_output")

            caps2_n_caps = 10
            caps2_n_dims = 16

    with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)):
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            init_sigma = 0.1
            W_init = tf.random_normal(
                shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims,
                       caps1_n_dims),
                stddev=init_sigma,
                dtype=tf.float32,
                name="W_init")

            # with tf.control_dependencies([W_init]):
            W = tf.get_variable(initializer=W_init, name="W", trainable=True)

    with tf.device(
            tf.DeviceSpec(device_type="GPU", device_index=device_index)):
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            W_tiled = tf.tile(
                W, [parallel_batch_size, 1, 1, 1, 1], name="W_tiled")

            caps1_output_expanded = tf.expand_dims(
                caps1_output, -1, name="caps1_output_expanded")
            caps1_output_tile = tf.expand_dims(
                caps1_output_expanded, 2, name="caps1_output_tile")
            caps1_output_tiled = tf.tile(
                caps1_output_tile, [1, 1, caps2_n_caps, 1, 1],
                name="caps1_output_tiled")
            # Computes Uj|i
            caps2_predicted = tf.matmul(
                W_tiled, caps1_output_tiled, name="caps2_predicted")
            # Bij's
            raw_weights = tf.zeros(
                [parallel_batch_size, caps1_n_caps, caps2_n_caps, 1, 1],
                dtype=tf.float32,
                name="raw_weights")
            # Computes the Cij's
            routing_weights = tf.nn.softmax(
                raw_weights, axis=2, name="routing_weights")

            weighted_predictions = tf.multiply(
                routing_weights, caps2_predicted, name="weighted_predictions")
            # computes the first Sjs
            weighted_sum = tf.reduce_sum(
                weighted_predictions,
                axis=1,
                keepdims=True,
                name="weighted_sum")

            # Computes the first Vjs
            caps2_output_round_1 = squash(
                weighted_sum, axis=-2, name="caps2_output_round_1")

            caps2_output_round_1_tiled = tf.tile(
                caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1],
                name="caps2_output_round_1_tiled")

            agreement = tf.matmul(
                caps2_predicted,
                caps2_output_round_1_tiled,
                transpose_a=True,
                name="agreement")
            raw_weights_round_2 = tf.add(
                raw_weights, agreement, name="raw_weights_round_2")
            routing_weights_round_2 = tf.nn.softmax(
                raw_weights_round_2, axis=2, name="routing_weights_round_2")
            weighted_predictions_round_2 = tf.multiply(
                routing_weights_round_2,
                caps2_predicted,
                name="weighted_predictions_round_2")
            weighted_sum_round_2 = tf.reduce_sum(
                weighted_predictions_round_2,
                axis=1,
                keepdims=True,
                name="weighted_sum_round_2")
            caps2_output_round_2 = squash(
                weighted_sum_round_2, axis=-2, name="caps2_output_round_2")
            # the following variable should have shape(None, 1, 10, 16, 1)
            # caps2_output = caps2_output_round_2
            #---------------------
            caps2_output_round_2_tiled = tf.tile(
                caps2_output_round_2, [1, caps1_n_caps, 1, 1, 1],
                name="caps2_output_round_2_tiled")

            agreement_2 = tf.matmul(
                caps2_predicted,
                caps2_output_round_2_tiled,
                transpose_a=True,
                name="agreement_2")
            raw_weights_round_3 = tf.add(
                raw_weights_round_2, agreement_2, name="raw_weights_round_3")
            routing_weights_round_3 = tf.nn.softmax(
                raw_weights_round_3, axis=2, name="routing_weights_round_3")
            weighted_predictions_round_3 = tf.multiply(
                routing_weights_round_3,
                caps2_predicted,
                name="weighted_predictions_round_3")
            weighted_sum_round_3 = tf.reduce_sum(
                weighted_predictions_round_3,
                axis=1,
                keepdims=True,
                name="weighted_sum_round_3")
            caps2_output_round_3 = squash(
                weighted_sum_round_3, axis=-2, name="caps2_output_round_3")
            caps2_output = caps2_output_round_3
            #---------------------

            y_prob = safe_norm(caps2_output, axis=-2, name="y_prob")
            y_prob_squeezed = tf.squeeze(
                y_prob, axis=-1, name="y_prob_squeezed")

            ###-----We wnt to store the lengths of vectors and their indices
            ###-----The length becomes a part of the loss function
            ###-----The index becomes the prediction

            # k represents how many digits we are looking for
            # gathers the top k longest vectors from the capsule layer and their indices
            top_k_capsules = tf.nn.top_k(
                y_prob_squeezed, k, name="top_k_capsules").indices
            y_pred = tf.squeeze(top_k_capsules, name="y_pred")

            S = tf.one_hot(y_pred, depth=caps2_n_caps)
            S = tf.reduce_sum(S, axis=1)

            y_pred = tf.cast(y_pred, tf.int64)

            # top_k_capsules.indices returns what numbers are predicted in the image
            # top_k_capsules.values returns the lengths of the vectors that represent the predicted numbers

            m_plus = 0.9
            m_minus = 0.1
            lambda_ = 0.5

            T = tf.one_hot(y, depth=caps2_n_caps, name="T")
            T = tf.reduce_sum(T, axis=1)

            caps2_output_norm = safe_norm(
                caps2_output,
                axis=-2,
                keep_dims=True,
                name="caps2_output_norm")
            present_error_raw = tf.square(
                tf.maximum(0., m_plus - caps2_output_norm),
                name="present_error_raw")
            present_error = tf.reshape(
                present_error_raw,
                shape=(-1, caps2_n_caps),
                name="present_error")

            absent_error_raw = tf.square(
                tf.maximum(0., caps2_output_norm - m_minus),
                name="absent_error")

            absent_error = tf.reshape(
                absent_error_raw,
                shape=(-1, caps2_n_caps),
                name="absent_error")

            L = tf.add(
                T * present_error,
                lambda_ * (1.0 - T) * absent_error,
                name="L")

            margin_loss = tf.reduce_mean(
                tf.reduce_sum(L, axis=1), name="margin_loss")

    with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)):
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            mask_with_labels = tf.placeholder_with_default(
                False, shape=(), name="mask_with_labels")

    with tf.device(
            tf.DeviceSpec(device_type="GPU", device_index=device_index)):
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            # reconstruction_targets = tf.cond(
            #     mask_with_labels,
            #     lambda: y,
            #     lambda: y_pred,
            #     name="reconstruction_targets")
            reconstruction_targets = y_pred

            reconstruction_mask = tf.one_hot(
                reconstruction_targets,
                depth=caps2_n_caps,
                name="reconstruction_mask")

            reconstruction_mask = tf.reduce_sum(reconstruction_mask, axis=1)

            reconstruction_mask_reshaped = tf.reshape(
                reconstruction_mask, [-1, 1, caps2_n_caps, 1, 1],
                name="reconstruction_mask_reshaped")

            caps2_output_masked = tf.multiply(
                caps2_output,
                reconstruction_mask_reshaped,
                name="caps2_output_masked")
            temp = tf.reshape(caps2_output_masked, [-1, 160])

    n_hidden1 = 512
    n_hidden2 = 1024
    n_hidden3 = 2048
    n_output = 28 * 56

    with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)):
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            decoder_input = tf.placeholder_with_default(
                input=temp, shape=[None, 160], name='decoder_input')

    with tf.device(
            tf.DeviceSpec(device_type="GPU", device_index=device_index)):
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            with tf.name_scope("decoder"):
                hidden1 = tf.layers.dense(
                    decoder_input,
                    n_hidden1,
                    activation=tf.nn.relu,
                    name="hidden1")
                hidden2 = tf.layers.dense(
                    hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
                hidden3 = tf.layers.dense(
                    hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3")
                decoder_output = tf.layers.dense(
                    hidden3,
                    n_output,
                    activation=tf.nn.sigmoid,
                    name="decoder_output")

            X_flat = tf.reshape(X, [-1, n_output], name="X_flat")
            squared_difference = tf.square(
                X_flat - decoder_output, name="squared_difference")
            reconstruction_loss = tf.reduce_mean(
                squared_difference, name="reconstruction_loss")

            alpha = 0.005
            loss = tf.add(
                margin_loss, alpha * reconstruction_loss, name="loss")

            reverse_y_pred = tf.reverse(y_pred, [-1], name="reverse_y_pred")
            correct_rev = tf.equal(reverse_y_pred, y, name="correct_rev")
            correct_reg = tf.equal(y_pred, y, name="correct_reg")
            correct_total = tf.logical_xor(
                correct_rev, correct_reg, name="correct_total")
            correct_total_one_hot = tf.cast(
                correct_total, tf.int32, name="correct_total_one_hot")
            correct_total_one_hot_float = tf.cast(
                correct_total_one_hot,
                tf.float32,
                name="correct_total_one_hot_float")
            correct_final = tf.reduce_mean(
                correct_total_one_hot_float, axis=-1, name="correct_final")
            accuracy = tf.reduce_mean(correct_final, name="accuracy")
    return loss


def make_parallel(fn, num_gpus, **kwargs):
    in_splits = {}
    for k, v in kwargs.items():
        in_splits[k] = tf.split(v, num_gpus)

    out_split = []
    for i in range(num_gpus):
        out_split.append(
            fn(num_gpus, i, **{k: v[i]
                               for k, v in in_splits.items()}))
    return tf.stack(out_split, axis=0)


def main(batch_size_train=BATCH_SIZE_TRAIN,
         train_length=TRAIN_LENGTH,
         validation_length=VALIDATION_LENGTH,
         test_length=TEST_LENGTH,
         num_epochs=NUM_EPOCHS,
         num_gpus=NUM_GPUS):

    (train_images, train_labels), (test_images,
                                   test_labels) = mnist.load_data()
    train, validation, test = generate_data_np(
        train_length,
        test_length,
        validation_length,
        train_images,
        train_labels,
        test_images,
        test_labels,
        batch_size_train,
        singles=False,
        doubles=True)

    # k represents how many digits we are looking for
    k = 2
    # Creates a placeholder for 28x56 images with one greyscale channel
    X = tf.placeholder(shape=[None, 28, 56, 1], dtype=tf.float32, name="X")
    # Creates a placeholder for the nxk labels
    y = tf.placeholder(shape=[None, k], dtype=tf.int64, name="y")

    loss = make_parallel(make_model, num_gpus, X=X, y=y)

    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(
        loss, colocate_gradients_with_ops=True, name="training_op")

    n_iterations_per_epoch = train_length // batch_size_train
    n_iterations_validation = validation_length // batch_size_train
    best_loss_val = np.infty

    # Saver object to save and restore the best model during training
    saver = tf.train.Saver()
    restore_checkpoint = True

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    export_dir = "{}/run-{}/".format('export_dir', now)
    checkpoint_path = "./checkpoints/my_capsule_network" + now

    # Builds restorable network information
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    input_tensor_info = tf.saved_model.utils.build_tensor_info(X)
    output_tensor_info = tf.saved_model.utils.build_tensor_info(y)

    masking_indices = tf.saved_model.utils.build_tensor_info(
        tf.get_default_graph().get_tensor_by_name("y_pred:0"))
    caps2_output_tensor_info = tf.saved_model.utils.build_tensor_info(
        tf.get_default_graph().get_tensor_by_name(
            "caps2_output_round_3/mul:0"))
    decoder_input_tensor_info = tf.saved_model.utils.build_tensor_info(
        tf.get_default_graph().get_tensor_by_name("decoder_input:0"))
    decoder_output_tensor_info = tf.saved_model.utils.build_tensor_info(
        tf.get_default_graph().get_tensor_by_name(
            "decoder/decoder_output/Sigmoid:0"))

    pred_signature = (tf.saved_model.signature_def_utils.build_signature_def(
        inputs={
            'X': input_tensor_info,
            'y': output_tensor_info,
            'decoder_input': decoder_input_tensor_info
        },
        outputs={
            'y_pred': masking_indices,
            'decoder_output': decoder_output_tensor_info,
            'caps2_output': caps2_output_tensor_info
        },
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    # Start timing total run time
    beginTime = time.time()
    with tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)) as sess:
        if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
            saver.restore(sess, checkpoint_path)
        else:
            init_global = tf.global_variables_initializer()
            sess.run(init_global)

        # Start timing total training time
        beginTrain = time.time()
        for epoch in range(num_epochs):
            train_accuracies = []
            train_losses = []
            # Start timing duration of epoch
            begin_epoch = time.time()
            for iteration in range(n_iterations_per_epoch):
                X_batch, y_batch = train.next_batch()
                feed_dict = {
                    X:
                    X_batch.reshape([-1, 28, 56, 1]),
                    y:
                    y_batch,
                    tf.get_default_graph().get_tensor_by_name("mask_with_labels:0"):
                    True
                }
                # Run the training operation and measure the loss:
                _, loss_train, acc_train = sess.run(
                    [
                        training_op, loss,
                        tf.get_default_graph().get_tensor_by_name("accuracy:0")
                    ],
                    feed_dict=feed_dict)
                train_accuracies.append(acc_train)
                train_losses.append(np.mean(loss_train, axis=0))
                print(
                    "\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                        iteration, (n_iterations_per_epoch + 1),
                        float(iteration * 100 / (n_iterations_per_epoch + 1)),
                        np.mean(train_losses)),
                    end="")

            # Resets the train data
            train.reset()

            # At the end of each epoch,
            # measure the validation loss and accuracy:
            loss_vals = []
            acc_vals = []
            for iteration in range(1, int(n_iterations_validation) + 1):
                X_batch, y_batch = validation.next_batch()
                loss_val, acc_val = sess.run(
                    [
                        loss,
                        tf.get_default_graph().get_tensor_by_name("accuracy:0")
                    ],
                    feed_dict={
                        X: X_batch.reshape([-1, 28, 56, 1]),
                        y: y_batch
                    })
                loss_val = sum(loss_val) / len(loss_val)
                loss_vals.append(loss_val)
                acc_vals.append(acc_val)
                print(
                    "\rEvaluating the model: {}/{} ({:.1f}%)".format(
                        iteration, n_iterations_validation,
                        float(iteration * 100 / n_iterations_validation)),
                    end=" " * 10)
            validation.reset()
            loss_val = np.mean(loss_vals)
            acc_val = np.mean(acc_vals)

            end_epoch = time.time()
            seconds = end_epoch - begin_epoch
            epoch_time = format_time(seconds)

            print(
                "\rEpoch: {} Training accuracy: {:.4f}%  Validation accuracy: {:.4f}%  Loss: {:.6f}{}  Duration: {}".
                format(epoch + 1,
                       np.mean(train_accuracies) * 100, acc_val * 100,
                       loss_val, " (improved)"
                       if loss_val < best_loss_val else "", epoch_time))

            # And save the model if it improved:
            if loss_val < best_loss_val:
                save_path = saver.save(sess, checkpoint_path)
                best_loss_val = loss_val

        endTrain = time.time()
        print('Total train time: {}'.format(
            format_time(endTrain - beginTrain)))

        builder.add_meta_graph_and_variables(
            sess, ["Post Training"],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                pred_signature
            })

        builder.save()

        sess.close()

    n_iterations_test = test_length // batch_size_train
    with tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)) as sess:
        saver.restore(sess, checkpoint_path)
        loss_tests = []
        acc_tests = []
        beginTest = time.time()
        for iteration in range(1, n_iterations_test + 1):
            X_batch, y_batch = test.next_batch()
            feed_dict = {X: X_batch.reshape([-1, 28, 56, 1]), y: y_batch}
            loss_test, acc_test = sess.run(
                [
                    loss,
                    tf.get_default_graph().get_tensor_by_name("accuracy:0")
                ],
                feed_dict=feed_dict)
            loss_test = sum(loss_test) / len(loss_test)
            loss_tests.append(loss_test)
            acc_tests.append(acc_test)
            print(
                "\rEvaluating the model: {}/{} ({:.1f}%)".format(
                    iteration, n_iterations_test,
                    iteration * 100 / n_iterations_test),
                end=" " * 10)
        loss_test = np.mean(loss_tests)
        acc_test = np.mean(acc_tests)
        endTest = time.time()
        print(
            "\rFinal test accuracy: {:.4f}% Loss: {:.6f} Duration: {}".format(
                acc_test * 100, loss_test, format_time(endTest - beginTest)))
        sess.close()

    endTime = time.time()
    print('Total run time: {}s'.format(format_time(endTime - beginTime)))


if __name__ == '__main__':
    main()
    print("IT WORKED!")
