


import  pandas as pd    
import random 
import math
import operator 
pd.set_option('display.max_columns',None)

data_file_name = "./user_taggedbookmarks-timestamps.dat"
data = pd.read_csv(data_file_name, sep='\t')

# 初始化
# 标签映射关系
user_tag_mapping = {}
items_tag_mapping = {}
user_item_mapping = {}
train_data = pd.DataFrame()
test_data = pd.DataFrame()
    
# data = data.head(n=10000) #取10000条做测试
# print('data.shape-->{}'.format(self.data.shape))
for index, row in data.iterrows():
    if random.random() < 0.2:
        test_data = test_data.append(row)
    else:
        train_data = train_data.append(row)
print('测试集train_data.shape-->:{}'.format(train_data.shape))
print('训练集test_data.shape-->:{}'.format(test_data.shape))
    
#加载标签 user_item_mapping
user_item_mapping = pd.DataFrame(train_data.groupby(['userID', 'bookmarkID'])['bookmarkID'].count())#.reset_index()
#重新命名bookmarkID的统计列为total
user_item_mapping.columns=['total']
user_item_mapping = user_item_mapping.reindex()

# user_tag_mapping
user_tag_mapping = pd.DataFrame(train_data.groupby(['userID', 'tagID'])['tagID'].count())#.reset_index()
user_tag_mapping.columns=['total']
user_tag_mapping = user_tag_mapping.reindex()

# items_tag_mapping
items_tag_mapping = pd.DataFrame(train_data.groupby(['bookmarkID', 'tagID'])['tagID'].count())#.reset_index()
items_tag_mapping.columns=['total']
items_tag_mapping = items_tag_mapping.reindex()
    

def recommend(userId, N=10, norm = 'SimpleTagBased'):
    uis = user_item_mapping.reset_index()#解决group by 生成的都是索引，把字段名字降下来,否则无法使用过滤器
    uts = user_tag_mapping.reset_index()#解决group by 生成的都是索引，把字段名字降下来,否则无法使用过滤器
    tis = items_tag_mapping.reset_index()
    # print(u_t)
    recommend_items = dict()
    # print(tis[tis.tagID == 1.0])
    # for j, t_i in tis[tis.tagID == 1.0].iterrows():
    #     print('t_i-->:{}'.format(t_i))
    for i, u_t in uts[uts['userID'] == userId].iterrows():
        u_t_tagID = u_t['tagID']
        # print('i:{}，tagID：{}'.format(i,u_t_tagID))
        for j, t_i in tis[tis['tagID'] == u_t_tagID].iterrows():
            norm_value = get_norm(norm,u_t,t_i)
            if t_i['bookmarkID'] not in recommend_items:
                recommend_items[t_i['bookmarkID']] = norm_value
            else:
                recommend_items[t_i['bookmarkID']] = recommend_items[t_i['bookmarkID']] + norm_value
    # print(recommend_items)
    return sorted(recommend_items.items(), key=operator.itemgetter(1), reverse=True)[0: N]

def get_norm(alg,u_t,t_i):
    # print('u_t,t_i-->:{}{}'.format(u_t, t_i))
    # SimpleTagBased算法
    norm_value = 0
    if alg == 'SimpleTagBased': 
        # print("SimpleTagBased is loaded...")
        # print('u_total,{};t_total,{}'.format(u_t['total'],t_i['total']))
        norm_value = u_t['total'] * t_i['total']
        # print('after_norm,{}'.format(norm))
    # NormTagBased算法
    elif alg == 'NormTagBased':
        print("NormTagBased is loaded...")
        norm_value = u_t['total'] / len(u_t['userID']) + t_i['total'] / len(t_i['tagID'])
    # TF-IDF算法
    elif alg == 'TF-IDF':
        print("TF-IDF is loaded...")
        norm_value = u_t['total'] / math.log(len(u_t['tagID'].items() + 1)) * t_i['total']
    return norm_value

def precisionAndRecall(N, alg = 'SimpleTagBased'):
    # print(test_data)
    #测试集分组
    test_data = test_data[['userID', 'bookmarkID']]
    # test_data = pd.DataFrame(test_data, columns=['userID','bookmarkID'])
    hit = 0 #命中个数
    h_callback = 0 #召回
    h_precision = 0#准确
    for user,group in test_data.groupby('userID'):
        # print('i-->:{},real_item-->:{}'.format(user,group))
        # user = row['userID']
        # item = row['bookmarkID']
        # print('user,{};是否在数据集,{}'.format(user,user in self.train_data['userID']))
        if user in train_data['userID']:
            rank = recommend(user, N, alg)
            # print('rank-->:{}'.format(rank))
            # user=8的时候
            # rank：[(14.0, 27.0), (11.0, 25.0), (61.0, 25.0), (8.0, 24.0), (9.0, 24.0)]
            for pre_items,rui in rank:
                items = group['bookmarkID'] #实际商品列表
                if pre_items in items:
                    # print('user:{};pre_items:{};items:{}'.format(user,pre_items,items))
                    hit = hit + 1
            h_callback = h_callback + len(items)
            h_precision = h_precision + N

    #初始化准确率和召回率
    accuracy_rate = 0
    recall_rate = 0
    if h_precision !=0:
        accuracy_rate = hit/(h_precision*1.0)
    if h_callback !=0:
        recall_rate = hit/(h_callback*1.0)
    return accuracy_rate,recall_rate

# 使用测试集，对推荐结果进行评估
def testRecommend(alg):
    #LSimpleTagBased
    print("{}推荐结果评估".format(alg))
    print("%3s %10s %10s" % ('N',"精确率",'召回率'))
    for n in [5, 10, 20, 40, 60, 80, 100]:
        precision_SimpleTagBased,recall_SimpleTagBased = precisionAndRecall(n, alg)
        print("%3d %10.3f%% %10.3f%%" % (n, precision_SimpleTagBased * 100, recall_SimpleTagBased * 100))
    return precision, recall

def algorithm_choose(type='SimpleTagBased'):
    precision, recall = testRecommend(type)
    return precion, recall


if __name__ == '__main__':
    # 数据加载
    
    # 训练集，测试集拆分，20%测试集
    train_test_split(0.2)
    # 使用模型给用户U推荐TopN的标签
    alg = 'TF-IDF'  # 'SimpleTagBased', 'NormTagBased', 'TF-IDF'
    # recommend(user=2, N=10, alg=alg)
     # 'SimpleTagBased', 'NormTagBased-1', 'NormTagBased-2', 'TagBased-TFIDF'
    testRecommend(alg)
    
    # 创建文件记录最后的结果
    score_file = np.random.randn(7, 9)
    score_file = pd.DataFrame(score_file)
    
    for index, type in enumerate(['SimpleTagBased', 'NormTagBased', 'TF-IDF']):
        N = [5, 10, 20, 40, 60, 80, 100] # 推荐数量
        print('计算{}模型的精确率和召回率'.format(type))
        precision, recall = algorithm_choose(type)
        print('{}模型计算完成'.format(type))
        for row in range(socre_file.shape[0]):
            score_file.iloc[row, 0] = N[row]
            score_file.iloc[row,index*2+1] =("%.3f%%" %(precision[row]*100))
        score_file.iloc[row,index*2+2] =("%.3f%%" %(recall[row]*100))
score_file.columns=["推荐数量N","NormTag1_精确率","NormTag1_召回率",
                    "NormTag2_精确率","NormTag2_召回率","Tag_TFIDF_精确率","Tag_TFIDF_召回率"]
score_file.to_csv(r"score.csv",encoding='gbk')
print(score_file)

# 算法成绩可视化
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
plt.rcParams['font.sans-serif'] = 'SimHei'

#s设置画布大小以及大标题、子图之间的距离
plt.figure(figsize=(8,6),dpi=72)
plt.suptitle("%10s" %('不同算法的推荐效果'), fontsize=20, fontweight='bold',color='green')
plt.subplots_adjust(top=0.85,wspace=0.1, hspace=0.5)
#开始画精确度
plt.subplot(2,1,1)
fmt='%.3f%%'
yticks=mtick.FormatStrFormatter(fmt)
plt.gca().yaxis.set_major_formatter(yticks) #修改副坐标将major改为minor即可
plt.title("precision",fontsize=18, fontweight='bold',color='blue')
plt.plot(score_file['推荐数量N'],[float(i.strip("%")) for i  in  score_file['SimpleTag_精确率']],label='SimpleTag1',linestyle="--")
plt.plot(score_file['推荐数量N'],[float(i.strip("%")) for i  in  score_file['NormTag_精确率']],label='NormTag',linestyle="--")
plt.plot(score_file['推荐数量N'],[float(i.strip("%")) for i  in  score_file['Tag_TFIDF_精确率']],label='Tag_TFIDF',linestyle="--")
plt.yticks([i  for  i  in  np.linspace(0,1.2,6)])
plt.xlabel("推荐商品数",fontsize=12, fontweight='bold',color='m')
plt.ylabel("精确率",fontsize=12, fontweight='bold',color='m')
plt.legend()
#开始画召回率
plt.subplot(2,1,2)
fmt='%.3f%%'
yticks=mtick.FormatStrFormatter(fmt)
plt.gca().yaxis.set_major_formatter(yticks) #修改副坐标将major改为minor即可
plt.title("recall",fontsize=18, fontweight='bold',color='blue')
plt.plot(score_file['推荐数量N'],[float(i.strip("%")) for i  in  score_file['SimpleTag_召回率']],label='SimpleTag',marker="o",markersize=4)
plt.plot(score_file['推荐数量N'],[float(i.strip("%")) for i  in  score_file['NormTag_召回率']],label='NormTag',marker="x",markersize=4)
plt.plot(score_file['推荐数量N'],[float(i.strip("%")) for i  in  score_file['Tag_TFIDF_召回率']],label='Tag_TFIDF',marker="*",markersize=4)

plt.xlabel("推荐商品数",fontsize=12, fontweight='bold',color='m')
plt.ylabel("召回率",fontsize=12, fontweight='bold',color='m')
plt.savefig(r"算法比较.jpg", dpi=72)
plt.legend()
plt.show()


    
    
