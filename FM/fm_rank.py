import numpy as np
from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm

# Read in data
# def loadData(filename,path="ml-100k/"):
def loadData(filename, path='E:/32.Project4CV/SVD4RS/ml-100k/'): 
    data = []
    y = []
    users=set()
    items=set()
    with open(path+filename) as f:
        for line in f:
            (user, movieid, rating, ts)=line.split('\t')
            data.append({ "user_id": str(user), "movie_id": str(movieid)})
            y.append(float(rating))
            users.add(user)
            items.add(movieid)

    return (data, np.array(y), users, items)

(train_data, y_train, train_users, train_items) = loadData("ua.base")
(test_data, y_test, test_users, test_items) = loadData("ua.test")

# 转换数据格式
v = DictVectorizer()
X_train = v.fit_transform(train_data)
X_test = v.transform(test_data)

# Build and train a Factorization Machine
fm = pylibfm.FM(num_factors=20, num_iter=50, verbose=True, task="regression", 
                initial_learning_rate=0.01, learning_rate_schedule="optimal")

fm.fit(X_train,y_train)

# Evaluate
preds = fm.predict(X_test)
from sklearn.metrics import mean_squared_error
print("FM MSE: %.4f" % mean_squared_error(y_test,preds))
#FM MSE: 0.9227
# 96.2%