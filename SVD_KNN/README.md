推荐系统中常用的算法是基于SVD的升级版：FunkSVD，BiasSVD，SVD++

### 3种算法相同点或异同点概括
* 3种方法的Loss计算公式均使用MSE+L2正则惩罚项。
时间复杂度：funkSVD、biasSVD，SVD++的时间复杂度越来越长，因为需要求解的参数越来越多。

* 3种SVD不计算预测值与空值的损失函数，计算预测值与之前存在原始值的损失函数
即：损失函数=P和Q矩阵乘积得到的评分，与实际用户评分之差。如果矩阵中有空值，y=Nan 通过SVD预测出y_hat 此时不能计算loss，因为y_hat-Nan没有意义。

* FunkSVD就是在SVD的技术上优化“数据稠密”+“计算复杂度告”+“只可以用来数据降维”难题的。一个矩阵做SVD分解成3个矩阵很耗时，同时还面临矩阵稀疏的问题，那么解决稀疏问题的同时只分解成两个矩阵。

* FunkSVD跟SVD相比只用两个矩阵P和Q相乘，奇异值通过开根号的方式分到奇异矩阵P和Q上，使用SGD求解P和Q让损失函数最小化，通过P和Q将矩阵补全，针对某个用户i查找缺失值的位置，按补全值从大到小进行推荐；

* BiasSVD把与用户、商品各自的偏好以及与个性化无关的用户考虑到优化目标函数，使用SGD求解4个变量，最终得到P和Q；

* SVD++把用户的隐式反馈（没有具体评分，但有可能点击、浏览，可看做一张表格，记为集合I(i)）修正值加入到优化目标函数。

## 详细介绍

### FunkSVD：SVD的特征值λ分摊到两个特征向量
（1）FunkSVD避开稀疏矩阵。SVD将矩阵拆解为：PSQ；funkSVD拆解矩阵为PQ。即特征值s通过开根号分给特征矩阵P和Q。
（2）FunkSVD需要设置k，求解topk个特征值，来对矩阵近似求解 （k是超参数 人工选定）
（3）FunkSVD优化方法使用SGD（随机梯度下降）的方法进行优化。SVD使用ALS使用交替最小二乘法（固定一个求解另一个），FunkSVD比传统SVD更适合做矩阵补全以及评分预测。

原版SVD: $A=PSQ^T$， FunkSVD: $A_(m×n)≈P_(m×k)^T Q_(k×n)$

任何矩阵都可按照 $ A=PΛQ^T $（PSQ）拆分，其中A=n*m,P=m*m,Λ=m*n,Q^T=n*n），基于对称矩阵AA^T和A^T A可以拆分为AA^T=PΛ_1 P^T 以及 A^T A=QΛ_1 Q^T，综合两者之后得到A=PΛ_1 Q^T。

最小化损失函数：MSE + L2正则
![image](https://user-images.githubusercontent.com/68730894/115145375-ca4aed00-a083-11eb-819c-0ee63c2323af.png)

* FunkSVD推荐思路
Step1，通过梯度下降法(SGD)，求解P和Q使得损失函数最小化
Step2，通过P和Q将矩阵A补全（学习）
Step3，针对某个用户u，查找之前值缺失的位置，按照补全值从大到小进行推荐

### BiasSVD：增加用户显示行为反馈
在FunkSVD的基础上，考虑了BaseLine算法，把用户偏好（比如乐观的用户打分偏高）、商品偏好（质量好的商品，打分偏高）、与个性化无关的内容作为偏好纳入loss损失函数进行优化，使用SGD方法是的矩阵P和Q最小化，针对用户i找到缺失值，按照补全值从大到小推荐。

BaseLine算法（b_ij=μ+b_i+b_j）与个性化无关的部分，因为没有考虑user2对Item3的打分，只考虑了用户本身对所有商品（item1、item2、item_n）的偏好。没有考虑user2对item3这个商品的偏好。FunkSVD设置为偏好(Bias)部分，考虑了user2对item3这个商品的偏好。通过优化函数中q_j^T p_i体现

![image](https://user-images.githubusercontent.com/68730894/115145415-f9f9f500-a083-11eb-9e96-c03f56b76295.png)

在优化函数中：预测值m_ij – 系统平均值μ – 用户偏好b_i – 商品偏好b_j – 预估用户对该商品的偏好q_j^T p_i。需要优化P、Q、 b_i和 b_j4个参数，而FunkSVD只优化P和Q 2个参数。

### SVD++：增加用户隐式行为反馈。
SVD++在BiasSVD算法（显式打分 ）基础上进行了改进，考虑用户的隐式反馈 。隐式反馈：没有具体的评分，但可能有点击，浏览等行为。显示反馈是用户u给商品i明确的打分。

需要优化的损失函数为：
![image](https://user-images.githubusercontent.com/68730894/115145427-0b430180-a084-11eb-8ceb-a3fecee2fde0.png)

对于某一个用户i，假设他的隐式反馈item集合为I(i)
用户i对商品j对应的隐式反馈修正值为c_ij

用户i所有的隐式反馈修正值之和为∑_(s∈N(i))▒c_sj 

在考虑用户隐式反馈和显示反馈的情况下，通过SGD优化得到P和Q。即：SVD++既有明确的打分，还有隐式的点击行为，用户的偏好、商品偏好，共同预测完成矩阵补全。



## SVD的基础补充
特征值分解是一个提取矩阵特征很不错的方法，但是它只是对方阵而言的，而奇异值分解是一个能适用于任意的矩阵的一种分解的方法。

特征向量之间一定线性无关，而且还正交（夹角=90°），余弦相似度为0。数学体现是：0.97760877*-0.54247681+0.21043072*0.84007078=0。

![image](https://user-images.githubusercontent.com/68730894/115144789-2eb87d00-a081-11eb-9057-d5d4be58dfd5.png)

P为左奇异矩阵，m*m维(里面的向量是正交的，U里面的向量称为左奇异向量)
Q^T(Q的转置)为右奇异矩阵，n*n维（里面的向量也是正交的，V里面的向量称为右奇异向量）

Λ对角线上的非零元素为特征值λ_1,λ_2,...,λ_k,k<min⁡(m,n)（除对角线外的元素都是0，对角线上的元素称为奇异值）, 对特征值λ排序，知道哪些特征重要哪些特征不重要。

![image](https://user-images.githubusercontent.com/68730894/115144801-3d069900-a081-11eb-9bfb-bfaf4dadbe5b.png)

![image](https://user-images.githubusercontent.com/68730894/115144853-7808cc80-a081-11eb-8fbb-18a25caf38c4.png)

```python 
from scipy.linalg import svd  
import numpy as np  
A = np.array([[1, 2], [1, 1], [0 ,0]])  
p, s, q = svd(A, full_matrices=False)  
print(‘P={}, S={}, Q={}’.format(p, s, q)) 
```

特征值召回top-K
如果对特征值λ排序，可以号回top-K个重要的特征向量。如：λ1为特征值（权重）， λ1> λ2> λ3>⋯> λn，λ1可作为主特征，λ可以理解为不同特征向量的权重。特征值λ保留的越多，越能还原矩阵A。此外，框中不同颜色代表不同特征对应的特征向量。
 
![image](https://user-images.githubusercontent.com/68730894/115144869-9242aa80-a081-11eb-9aea-c302cc3bef6e.png)



SVD矩阵分解可以使用少量的特征代表整个矩阵。下图对一张图片（矩阵A）进行特征提取，保留大约10%的特征向量（特征值λ=10%）就能与原图的效果差不多，大幅压缩了时空复杂度。


```python 
取top-k个λ值  
def get_topk_feature(s, k):  
   # 对于matrix，保留前k个，k维embedding  
   s_temp = np.zeros(s.shape[0])  
   s_temp[0:k] = s[0:k]  
   s = s_temp * np.identity(s.shape[0])  
```

### 应用场景
* 推荐系统：user相当于左奇异矩阵，item相当于右奇异矩阵，特征值排序后取top-N
* 特征工程Embedding：稀疏矩阵变稠密矩阵
* 矩阵补全：可以接受大量非零样本（空值）,实际上传统SVD更适合做降维（和填充原矩阵空值=预测评分），适用于时间序列缺失值补全（传感器收集数据存在大量缺失值）
* SVD可以应用于寻找大量数据的隐含“模式”（也就是Embedding），如模式识别，数据压缩，图片压缩，推荐系统

* 评分预测:传统SVD分解要求矩阵是稠密的 => 矩阵中的元素不能有缺失



