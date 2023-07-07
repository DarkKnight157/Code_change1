from __future__ import print_function
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)

class LayerNormalization(Layer):
    def __init__(self, epsilon=1e-8, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer='ones', trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer='zeros', trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.var(inputs, axis=-1, keepdims=True)
        normed = (inputs - mean) / K.sqrt(variance + self.epsilon)
        return self.gamma * normed + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape
