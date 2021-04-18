import os
# os.environ["CUDA_DEVICE_ORDER"] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
from sklearn import preprocessing

learning_rate = 0.1
is_train = True

class Model(object):
    def __init__(self, num_users, num_items, k):
        #用户类型数
        self.num_users = num_users
        #物品数
        self.num_items = num_items
        #特征数
        self.k = k
        #用户感兴趣物品类型特征embedding
        self.feature_user = tf.Variable(tf.random.normal([num_users, k]))  # 生成10*1的张量
        #物品类型特征embedding
        self.feature_item = tf.Variable(tf.random.normal([num_items, k]))
        self.ckpt = self.makeCheckpoint()
        
    def __call__(self, user_ids, item_ids):
        embbeding_u = tf.nn.embedding_lookup(self.feature_user, list(np.array(user_ids)-1))
        embbeding_i = tf.nn.embedding_lookup(self.feature_item, list(np.array(item_ids)-1))
        #user_feature*item_feature转置，获取用户与某个物品的评价（对角线上的值）
        rt_ui = tf.linalg.diag_part(tf.matmul(embbeding_u, embbeding_i, transpose_b=True))
        return rt_ui

    def get_user_embbeding(self, user_ids):
        embbeding_u = tf.nn.embedding_lookup(self.feature_user, list(np.array(user_ids)-1))
        return embbeding_u

    def get_item_embbeding(self, item_ids):
        embbeding_i = tf.nn.embedding_lookup(self.feature_item, list(np.array(item_ids)-1))
        return embbeding_i

    def makeCheckpoint(self):
        return tf.train.Checkpoint(
            feature_user=self.feature_user,
            feature_item=self.feature_item)

    def saveVariables(self):
        self.ckpt.save('./save/ckpt')

    def restoreVariables(self):
        #恢复self.feature_user和self.feature_item值
        status = self.ckpt.restore(tf.train.latest_checkpoint('./save'))
        status.assert_consumed()  # Optional check

def loss(predicted_y, desired_y, embbeding_u, embbeding_i, ld):
    plos = tf.reduce_mean(tf.square(predicted_y - desired_y))
    plos_w = tf.reduce_mean(ld*tf.multiply(embbeding_u, embbeding_u)) + tf.reduce_mean(ld*tf.multiply(embbeding_i, embbeding_i))
    plos = plos + plos_w
    return plos

# 定义优化器
optimizer = tf.optimizers.SGD(learning_rate)

def train(model, inputs_user_ids, inputs_item_ids, outputs, idx):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs_user_ids, inputs_item_ids), outputs,
                            model.get_user_embbeding(inputs_user_ids),
                            model.get_item_embbeding(inputs_item_ids), 0.001)
    if idx % 100000 == 0:
        print('idx:%d, loss:%f' % (idx, current_loss))

    gradients = t.gradient(current_loss, [model.feature_user, model.feature_item])
    optimizer.apply_gradients(zip(gradients, [model.feature_user, model.feature_item]))



df = pd.read_csv('E:/32.Project4CV/SVD4RS/ml-1m/ratings.dat', sep='::', names=['user', 'item', 'rating', 'timestamp'], header=None)
df = df.drop('timestamp', axis=1)

# 开始数据处理
num_items = df.item.nunique()
num_users = df.user.nunique()
print("USERS: {} ITEMS: {}".format(num_users, num_items))

r = df['rating'].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(r.reshape(-1,1))
df_normalized = pd.DataFrame(x_scaled)
df['rating'] = df_normalized

user_ids = df['user'].values.astype(int)
item_ids = df['item'].values.astype(int)
outputs = df['rating'].values.astype(float)

print('user总数{},商品总数{}, 评分总数{}'.format(len(user_ids), len(item_ids), len(outputs)))
#最大用户id
max_user_id = np.amax(user_ids)
#最大电影id
max_item_id = np.amax(item_ids)

model = Model(max_user_id, max_item_id, 20) # 20维Embedding

if is_train:
    print(model.feature_user[:10])

    max_len = len(user_ids)
    batch_size = 2000
    epochs = range(20)
    for epoch in epochs:
        cur_idx = 0
        print('Epoch %2d' % (epoch))
        while True:
            start_idx = cur_idx
            end_idx = cur_idx + batch_size
            if start_idx >= max_len:
                break
            if end_idx > max_len:
                end_idx = max_len

            cur_user_ids = user_ids[start_idx:end_idx]
            cur_item_ids = item_ids[start_idx:end_idx]
            cur_outputs = outputs[start_idx:end_idx]

            train(model, cur_user_ids, cur_item_ids, cur_outputs, cur_idx)
            cur_idx += batch_size

        print(model.feature_user[0:1])
        #保存权重
        model.saveVariables()
else:
    model.restoreVariables() #读取权重

print('===========================================')
u1rate = tf.squeeze(tf.matmul(model.feature_user[0:1], model.feature_item, transpose_b=True))
print(u1rate);
# 获取最大值的前10个物品
print(tf.argsort(u1rate, direction='DESCENDING')[0:10])
print('===========================================')

u1rate = tf.squeeze(tf.matmul(model.feature_user[1:2], model.feature_item, transpose_b=True))
print(u1rate);
# 获取最大值的前10个物品
np.set_printoptions(threshold=sys.maxsize)
print(tf.argsort(u1rate, direction='DESCENDING')[0:10].numpy())
print('===========================================')