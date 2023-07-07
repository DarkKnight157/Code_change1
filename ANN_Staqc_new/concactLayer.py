import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

# 设置随机种子
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)

# 定义 ConcatLayer 类
class ConcatLayer(Layer):
    def __init__(self,**kwargs):
        super(ConcatLayer, self).__init__(**kwargs)

    # 定义 call 函数
    def call(self, inputs, **kwargs):
        '''
        将输入张量在第二维度上切割，并在第三维度上进行拼接
        axis = 1 表示在第二维度上进行切割
        去除张量的第一维度,将其变为(batch_size, n_features*block_size)
        注意在这里我们默认 input 的第一维度是 batch_size
        '''
        block_level_code_output = tf.concat(tf.split(inputs, inputs.shape[1], axis=1), axis=2)  
        block_level_code_output = tf.squeeze(block_level_code_output, axis=1)  
        print(block_level_code_output) 
        return block_level_code_output

    # 定义 compute_output_shape 函数
    def compute_output_shape(self, input_shape):
        '''
        返回结果的形状为(batch_size, n_features*block_size)
        '''
        print("===========================",input_shape)
        
        return (input_shape[0], input_shape[1]*input_shape[2])
