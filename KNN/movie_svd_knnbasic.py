
from surprise.model_selection import cross_validate, KFold
import surprise
import pandas as pd
import time

from surprise.prediction_algorithms.knns import KNNBaseline


time1 = time.time()

def get_model(data, model='FunkSVD'):
    print('使用{}模型进行推荐'.format(model))
    # 使用funkSVD
    if model == 'FunkSVD': 
        algo = surprise.SVD(biased=False, n_factors=20, n_epochs=20)
        # 正则化参数: reg_bu = reg_pu = 0.02
    if model == 'BiasSVD':
        algo = surprise.SVD(biased=True, n_factors=20, n_epochs=20)
        # # 正则化参数: reg_bu = reg_bi = reg_pu = reg_qi = 0.02
    elif model == 'SVD++':
        algo = surprise.SVDpp(n_factors=20, n_epochs=20)
        # 正则化参数: reg_bu = reg_bi = reg_pu = reg_qi = reg_yj = 0.02
    elif model == 'KNNBaseline':    
        sim_options = {'name': 'pearson_baseline', 'user_based': False}
        algo = KNNBaseline(sim_options=sim_options)
    elif model == 'SurpriseBaseline': # 给定user和item，r_ui=bui=μ+bu+bi
        bsl_options = {'method': 'als','n_epochs': 20,'reg_u': 12,'reg_i': 5} # 也可使用SGD
        algo = surprise.BaselineOnly(bsl_options=bsl_options)
    elif model == 'NormalPredictor': # 对输入数据进行规范化后的算法
        # prediction r^ui is generated from a normal distribution N(μ^,σ^2) where μ^ and σ^ are estimated from the training data using Maximum Likelihood Estimation.
        algo = surprise.NormalPredictor()
    elif model == 'KNNBasic':     # UserCF or  ItemCF
        algo = surprise.KNNBasic(k=20, min_k=5, sim_options={'user_based':True}, verbose=True)
    # 协同过滤
    elif model == 'userCF':
        # 取最相似的用户计算时，只取最相似的k个
        algo = surprise.KNNWithMeans(k=20, min_k=1, sim_options={'user_based': True, 'verbose': 'True'})
    elif model == 'itemCF':
        algo = surprise.KNNWithMeans(k=20, min_k=1, sim_options={'user_based':False, 'verbose':'True'})

    # 定义K折交叉验证迭代器，K=3
    kf = KFold(n_splits=3, random_state=2021, shuffle=True)
    RMSE_loss = []
    for trainset, testset in kf.split(data):
        # 训练并预测
        algo.fit(trainset)
        predictions = algo.test(testset)
        # 计算RMSE
        rmse = surprise.accuracy.rmse(predictions, verbose=True)
        RMSE_loss.append(rmse)
    mean_loss = sum(RMSE_loss) / len(RMSE_loss)
    print('算法{}的平均RMSE Loss为{}'.format(model, mean_loss))
        
    print('花费多长时间', time.time() - time1)
    return algo
"""
    计算Top-N
"""
def calculate_topN(model, data, topN=10):  
    # 创立字典user_items，记录user用过哪些movie
    user_items = {}
    # 创建userID和itemID，分别记录所有的user和item
    userID, itemID = [], []
    
    # 加载数据
    # reader = surprise.Reader(line_format='user item rating', sep=',', skip_lines=1)
    # data = surprise.Dataset.load_from_df(file_path, reader)
    # data = data.build_full_trainset()
    
    # 遍历数据集data，生成user_items列表，储存用户已经评分过的items
    data = data.build_full_trainset() # 使用代码为了data.all_ratings()出错
    for user, item, rating in data.all_ratings():   
        user_items.setdefault(user, [])
        user_items[user].append(item)
        
        if item not in itemID:   
            itemID.append(item)
    
    userID = list(user_items.keys())
    
    # top-N 推荐
    # 遍历数据集，找出user没有购买过的商品
    # 用user_newitems_ratings列表记录对没有购买过的商品的预估评分
    user_newitems_ratings = {}
    
    # 导入算法搭建模型
    # algo = surprise.SVD(n_factors=30).fit(data)
    # 之前有已经训练好的模型algo
    
    # 遍历所有可能的用户-物品对，找出用户没有购买过的商品
    for user in userID:    
        user_newitems_ratings.setdefault(user, {})
        
        for item in itemID:   
            if item not in user_items[user]:   # user没有购买过的商品
                user_newitems_ratings[user][item] = model.predict(user, item, verbose=False).est
    
    # 对用户评分进行排序
    for user in user_newitems_ratings.keys():   
        user_newitems_ratings[user] = sorted(user_newitems_ratings[user].items(), key=lambda x: x[1], reverse=True)
        
    # 获取user的topN推荐
    return user_newitems_ratings[user][:topN]

def uid4iid(model, uid, iid, r_ui=4):
    # 得到uid和iid之间是否相关
    pred = model.predict(uid, iid, r_ui, verbose=True)
    return pred

def load_data(file_path):    
    reader = surprise.Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
    data = surprise.Dataset.load_from_file(file_path, reader=reader)
    # train_data = data.build_full_trainset()
    return data

if __name__ == "__main__": 
    train_set = load_data('ml-1m/ratings.csv')
    # 基于协同过滤的推荐系统
    # 'KNNBaseline', 'SurpriseBaseline','NormalPredictor', 'KNNBasic'
    # 协同过滤：'itemCF'、'userCF'
    model = get_model(train_set, model='SurpriseBaseline') 
    recommand_result = calculate_topN(model=model,data=train_set, topN=5)
    print(recommand_result)
    print('--测试给定用户id和商品id，他们之间是否有相关性---')
    relationship = uid4iid(model, uid=138, iid=243, r_ui=4)
    
    
