import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

# 设定随机数种子
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)


class AttentionLayer(Layer):
    # 初始化函数
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    # 构建函数
    def build(self, input_shape):
        # 检查输入是否正确
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('AttentionLayer应该接受2个输入')
        if not input_shape[0][-1] == input_shape[1][-1]:
            raise ValueError('嵌入的大小应该相同')

        self.kernel = self.add_weight(shape=(input_shape[0][-1], input_shape[0][-1]),
                                      initializer='glorot_uniform',
                                      name='kernel',
                                      trainable=True)

        super(AttentionLayer, self).build(input_shape)
    # 调用函数
    def call(self, inputs):
        a = K.dot(inputs[0], self.kernel)
        y_trans = K.permute_dimensions(inputs[1], (0, 2, 1))
        b = K.batch_dot(a, y_trans, axes=[2, 1])
        return K.tanh(b)

    # 输出维度计算函数
    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][1], input_shape[1][1])
