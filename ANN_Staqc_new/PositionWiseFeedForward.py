from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow as tf
import os
import numpy as np
import random


class PositionWiseFeedForward(Layer):
    # inner_dim 隐藏层的维度，一般默认2048,model_dim是词向量的维度
    def __init__(self, model_dim, inner_dim, trainable=True, **kwargs):
        self.model_dim = model_dim
        self.inner_dim = inner_dim
        self.trainable = trainable
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.weights_inner = self.add_weight(
            shape=(input_shape[-1], self.inner_dim),
            initializer='glorot_uniform',
            trainable=self.trainable,
            name="weights_inner")
        self.weights_out = self.add_weight(
            shape=(self.inner_dim, self.model_dim),
            initializer='glorot_uniform',
            trainable=self.trainable,
            name="weights_out")
        self.bais_inner = self.add_weight(
            shape=(self.inner_dim,),
            initializer='uniform',
            trainable=self.trainable,
            name="bais_inner")
        self.bais_out = self.add_weight(
            shape=(self.model_dim,),
            initializer='uniform',
            trainable=self.trainable,
            name="bais_out")

        super().build(input_shape)

    def call(self, inputs):
        if K.dtype(inputs) != 'float32':
            inputs = K.cast(inputs, 'float32')
        inner_out = K.relu(K.dot(inputs, self.weights_inner) + self.bais_inner)
        outputs = K.dot(inner_out, self.weights_out) + self.bais_out
        print("==", outputs.shape)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape
