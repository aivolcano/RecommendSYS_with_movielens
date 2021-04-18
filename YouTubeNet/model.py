#-*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
# from tensorflow.keras.layers import Input, Embedding, concatenate, Dense, Dropout
import tensorflow.keras.backend as K
from SequencePoolingLayer import SequencePoolingLayer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# self-attention
class Self_Attention(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        #inputs_shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
            shape=(3,input_shape[2], self.output_dim),
            initializer='uniform',
            trainable=True)

        super(Self_Attention, self).build(input_shape)

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])

        #print("WQ.shape", WQ.shape)
        #print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(WK, [0, 2, 1]).shape)

        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))
        QK = QK / (64**0.5)
        QK = tf.nn.softmax(QK)
        #print("QK.shape", QK.shape)

        V = K.batch_dot(QK, WV)
        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

def YouTubeNet(
    sparse_input_length=1,
    dense_input_length=1,
    sparse_seq_input_length=50,
    
    embedding_dim = 64,
    neg_sample_num = 10,  # 正样本 ：负样本 = 1：10
    user_hidden_unit_list = [128, 64],
    # item_hidden_unit_list = [128, 64],
    ):

    # 1. 输入层Input layer
    user_id_input_layer = tf.keras.Input(shape=(sparse_input_length, ), name="user_id_input_layer") # (None,1)
    gender_input_layer = tf.keras.Input(shape=(sparse_input_length, ), name="gender_input_layer") # (None,1)
    age_input_layer = tf.keras.Input(shape=(sparse_input_length, ), name="age_input_layer") # (None,1)
    occupation_input_layer = tf.keras.Input(shape=(sparse_input_length, ), name="occupation_input_layer") # (None,1)
    zip_input_layer = tf.keras.Input(shape=(sparse_input_length, ), name="zip_input_layer") # (None,1)

    user_click_item_seq_input_layer = tf.keras.Input(shape=(sparse_seq_input_length, ), # (None, 50)
                                                     name="user_click_item_seq_input_layer")
    user_click_item_seq_length_input_layer = tf.keras.Input(shape=(sparse_input_length, ), #(none, 1)
                                                            name="user_click_item_seq_length_input_layer")
    
    
    pos_item_sample_input_layer = tf.keras.Input(shape=(sparse_input_length, ), #(none, 1)
                                                 name="pos_item_sample_input_layer")
    neg_item_sample_input_layer = tf.keras.Input(shape=(neg_sample_num, ), #(none, 10)
                                                 name="neg_item_sample_input_layer")
    
    # 2. Embedding layer
    # (None, 1) --> (None, 1, 64)   [batch_size, seq_len, embed_dim]
    user_id_embedding_layer = tf.keras.layers.Embedding(6040+1, embedding_dim, # embedding_dim = int(np.sqrt(6040 + 1))
                                                        mask_zero=True,
                                                        name='user_id_embedding_layer')(user_id_input_layer)
    # (None, 1) --> (None, 1)
    gender_embedding_layer = tf.keras.layers.Embedding(2+1, embedding_dim, #embedding_dim = int(np.sqrt(2 + 1))
                                                       mask_zero=True,
                                                       name='gender_embedding_layer')(gender_input_layer)
    # (None, 1) --> (None, 1, 64)
    age_embedding_layer = tf.keras.layers.Embedding(7+1, embedding_dim, # embedding_dim = int(np.sqrt(7 + 1))
                                                    mask_zero=True,
                                                    name='age_embedding_layer')(age_input_layer)
    # (None, 1) --> (None, 1)
    occupation_embedding_layer = tf.keras.layers.Embedding(21+1, embedding_dim, # embedding_dim = int(np.sqrt(21+1))
                                                           mask_zero=True,
                                                           name='occupation_embedding_layer')(occupation_input_layer)
    # (None, 1) --> (None, 1)
    zip_embedding_layer = tf.keras.layers.Embedding(3439+1, embedding_dim,# embedding_dim = int(np.sqrt(3439+1))
                                                    mask_zero=True,
                                                    name='zip_embedding_layer')(zip_input_layer)

    # (None, 1), 正样本(None,1), 负样本(None, 10) --> (None, 1)
    item_id_embedding_layer = tf.keras.layers.Embedding(3706+1, embedding_dim,
                                                        mask_zero=True,
                                                        name='item_id_embedding_layer')
    # (None, 1) --> (None, 1)
    pos_item_sample_embedding_layer = item_id_embedding_layer(pos_item_sample_input_layer)
    # (None, 10) --> (None, 10)
    neg_item_sample_embedding_layer = item_id_embedding_layer(neg_item_sample_input_layer)
    
    user_click_item_seq_embedding_layer = item_id_embedding_layer(user_click_item_seq_input_layer)
    # user_click_item_seq_embedding_layer = SequencePoolingLayer(sequence_mask_length=sparse_seq_input_length)\
    #    ([user_click_item_seq_embedding_layer, user_click_item_seq_length_input_layer])

    # SumPooling，把每部电影的Embedding求均值，得到用户的平均喜好。此时每部电影embedding的权重是 1/n, n是人工规则设置几部电影
    user_click_item_seq_embedding_layer = tf.reduce_mean(user_click_item_seq_embedding_layer, 1, keepdims=True)


    ### ********** ###
    # user part
    ### ********** ###

    # 3. Concat "sparse" embedding & "sparse_seq" embedding
    # (None, 1, 64) --> (None, 1, 384) # 384 = 6 * 64
    # 每个特征的权重是 1/n 可以进行加权，赋予每个部分不同的权重
    # user_embed_w = tf.Variable([1, 1, 1], trainable=True)
    user_embedding_layer = tf.keras.layers.concatenate([user_id_embedding_layer, gender_embedding_layer, age_embedding_layer,
                                       occupation_embedding_layer, zip_embedding_layer, user_click_item_seq_embedding_layer],
                                       axis=-1)
    # Attention加权
    user_embedding_layer = Self_Attention(output_dim=128)(user_embedding_layer)

    # 喂给全连接层 最后输出 # (None, 1, 64)
    for i, u in enumerate(user_hidden_unit_list):
        user_embedding_layer = tf.keras.layers.Dense(u, activation="relu", name="itemFC_{0}".format(i+1))(user_embedding_layer)
        user_embedding_layer = tf.keras.layers.Dropout(0.3)(user_embedding_layer)
        user_embedding_layer = tf.keras.layers.BatchNormalization()(user_embedding_layer)

    
    ### ********** ###
    # item part
    ### ********** ###
    # (None, 1, 64) + (None, 10, 64) --> (None, 11, 64)
    item_embedding_layer = tf.keras.layers.concatenate([pos_item_sample_embedding_layer, neg_item_sample_embedding_layer], \
                                       axis=1)
    # 使用全连接层进行学习
    # for i, u in enumerate(item_hidden_unit_list):
    #     item_embedding_layer = tf.keras.layers.Dense(u, activation='relu', name="userFC_{0}".format(i+1))(item_embedding_layer)
    #     item_embedding_layer = tf.keras.layers.Dropout(0.3)(item_embedding_layer)
    #     item_embedding_layer = tf.keras.layers.BatchNormalization()(item_embedding_layer)

    # (None, 11, 64) --> (None, 64, 11)
    item_embedding_layer = tf.transpose(item_embedding_layer, [0, 2,1])


    # Output
    # (None, 1, 64) * (None, 64, 11) = (None, 1, 11)
    # user 和 item的相似度：user_embedding_layer * item_embedding_layer
    dot_output = tf.matmul(user_embedding_layer, item_embedding_layer) 
    dot_output = tf.nn.softmax(dot_output) # 输出11个值，index为0的值是正样本，负样本的索引位置为[1-10]

    user_inputs_list = [user_id_input_layer, gender_input_layer, age_input_layer, \
                        occupation_input_layer, zip_input_layer, \
                        user_click_item_seq_input_layer, user_click_item_seq_length_input_layer]
    
    item_inputs_list = [pos_item_sample_input_layer, neg_item_sample_input_layer]

    model = tf.keras.models.Model(inputs = user_inputs_list + item_inputs_list,
                  outputs = dot_output)
    
    print(model.summary())
    tf.keras.utils.plot_model(model, to_file='YouTubeNet_model.png', show_shapes=True)



    model.__setattr__("user_input", user_inputs_list)
    model.__setattr__("user_embedding", user_embedding_layer)
    
    model.__setattr__("item_input", pos_item_sample_input_layer)
    model.__setattr__("item_embedding", pos_item_sample_embedding_layer)
    
    return model
