import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)


class PositionEmbedding(Layer):
    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size if size%2==0 else size+1  # 大小必须是偶数
        self.mode = mode
        super(PositionEmbedding, self).__init__(**kwargs)

    def call(self, x):  # 上一层一般就是embedding层，batch_size,seq_len,model_dim
        if (self.size is None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])  # d_model的长度,比如512
        
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]  
        position_j = 1. / (10000 ** (2 * K.arange(self.size / 2, dtype='float32') / self.size))
        position_j = K.expand_dims(position_j, axis=0)
        position_i = K.arange(seq_len, dtype='float32')
        position_i = K.expand_dims(position_i, axis=1)
        position_ij = K.dot(position_i, position_j)
        position_ij_2i = K.sin(position_ij[:, :, ::2])[..., tf.newaxis]
        position_ij_2i_1 = K.cos(position_ij[:, :, 1::2])[..., tf.newaxis]
        position_ij = K.concatenate([position_ij_2i, position_ij_2i_1], axis=-1)
        position_ij = K.reshape(position_ij, (batch_size, seq_len, self.size))
        
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], axis=-1)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)
