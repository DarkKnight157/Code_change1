import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, backend as K

# 设置随机种子
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)


class SelfAttention(layers.Layer):
    def __init__(self, r, da, name, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.r = r
        self.da = da
        self.scope = name

    def build(self, input_shape):
        self.Ws1 = self.add_weight(name='Ws1_' + self.scope,
                                   shape=(input_shape[-1], self.da),
                                   initializer='glorot_uniform',
                                   trainable=True)
        self.Ws2 = self.add_weight(name='Ws2_' + self.scope,
                                   shape=(self.da, self.r),
                                   initializer='glorot_uniform',
                                   trainable=True)

    def call(self, inputs, **kwargs):
        A1 = K.dot(inputs, self.Ws1)
        A1 = tf.tanh(tf.transpose(A1, [0, 2, 1]))
        A1 = tf.transpose(A1, [0, 2, 1])
        A_T = K.softmax(K.dot(A1, self.Ws2))
        A = K.permute_dimensions(A_T, (0, 2, 1))
        B = tf.matmul(A, inputs)
        tile_eye = tf.tile(tf.eye(self.r), [tf.shape(inputs)[0], 1])
        tile_eye = tf.reshape(tile_eye, [-1, self.r, self.r])
        AA_T = tf.matmul(A, A_T) - tile_eye
        P = tf.square(tf.norm(AA_T, axis=[-2, -1], ord='fro'))
        return [B, P]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[2], self.r), (input_shape[0],)]


# 测试
inputs = tf.keras.Input(shape=(4, 150))
outputs = SelfAttention(r=3, da=120, name='test')(inputs)
model = tf.keras.Model(inputs, outputs)
model.summary()
