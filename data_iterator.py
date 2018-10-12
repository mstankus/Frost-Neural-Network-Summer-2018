import numpy as np
from keras.datasets import mnist


class Data_np(object):
    def __init__(self, data, batch_size):
        self.data = data
        # keeps track of where we are in data set so we know what elements
        # to give when prompted
        self.data_set_marker = 0
        self.batch_size = batch_size

    def next_batch(self):
        front = self.data_set_marker
        tail = self.data_set_marker + self.batch_size
        self.data_set_marker = tail
        return (self.data[0][front:tail], self.data[1][front:tail])

    # call before the next epoch begins
    def reset(self):
        self.data_set_marker = 0
