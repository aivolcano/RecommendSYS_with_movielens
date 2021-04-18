import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from deepctr.feature_column import SparseFeat, get_feature_names
from deepctr.models import NFM, DeepFM
from sklearn.model_selection import train_test_split

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 加载数据
data = pd.read_csv('E:/32.Project4CV/SVD4RS/movielens-1m.csv')
sparse_features = ['movie_id','user_id','gender','age','occupation','zip']
target = ['rating']

# 对特征编码
for feature in sparse_features:
    lbe = LabelEncoder()
    data[feature] = lbe.fit_transform(data[feature])
    
# 计算每个特征中的不同特征值个数
fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique()) for feat in sparse_features]
linear_feature_columns = fixlen_feature_columns
dnn_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)


# 切分数据集
train, test = train_test_split(data, test_size = 0.2)
train_model_input = {name:train[name].values for name in feature_names}
test_model_input = {name:test[name].values for name in feature_names}
# 使用NFM进行训练
model = NFM(linear_feature_columns, dnn_feature_columns, task='regression', dnn_hidden_units=(256, 128, 64))
# model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression', dnn_hidden_units=(256, 128, 64)) # dnn_dropout=0.1,

model.compile('rmsprop',loss='mse',metrics=['mse'])
history = model.fit(train_model_input, train[target].values, batch_size=1500, epochs=10, verbose=True, validation_split=0.1)
# epoch=80
# 使用NFM进行预测
pred_ans = model.predict(test_model_input, batch_size=256)
rmse = mean_squared_error(test[target].values, pred_ans, squared=False)
print('test RMSE', rmse)
# 输出RMSE或者MSE

