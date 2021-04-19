# RecommendSYS
build the ResSys with FunkSVD, FM, itemCF/UserCF, wide&amp;deep with residual net, deepFM with residual net and etc. I try to collect all the algorithm as soon as possible.


### 基于标签的推荐系统：基于流行度的推荐，热门商品可降权
点评网站，如：大众点评、豆瓣，用户给某个商品打分，系统会自动提示一些标签，此时我们可以设计的推荐算法及策略有哪些？
 * (1) 给用户u推荐整个系统最热门的标签;
 * (2) 给用户u推荐物品i上最热门的标签;
 * (3) 给用户u推荐他自己经常使用的标签;
 * (4) 将方法2和3进行加权融合，生成最终的标签推荐结果。

* SVD: 
  * FunkSVD:
  * BiasSVD:
  * SVD++

* 因子分解机：FM

* 协同过滤

* 深度学习（tensorflow 2.x）
wide & deep

deepfm

NFM
