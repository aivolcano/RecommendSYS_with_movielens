
### 基于标签的推荐系统
### 任务目标：开发基于词频的推荐算法
基于流行度的推荐，热门商品可降权



### 点评网站，如：大众点评、豆瓣，用户给某个商品打分，系统会自动提示一些标签，此时我们可以设计的推荐算法及策略有哪些？
(1) 给用户u推荐整个系统最热门的标签;

(2) 给用户u推荐物品i上最热门的标签;

(3) 给用户u推荐他自己经常使用的标签;

(4) 将方法2和3进行加权融合，生成最终的标签推荐结果。


### SingleTagBased算法
(1) 统计每个用户的常用标签;
(2) 对每个标签，统计被打过这个标签次数最多的商品;
(3) 对于一个用户，找到他常用的标签，然后找到具有这些标签的最热门物品推荐给他;
(4) 用户u对商品i的兴趣

![image](https://user-images.githubusercontent.com/68730894/115217049-ae088800-a137-11eb-8ebc-f30e17ecb592.png)


### TagBased-TFIDF算法
一个Tag高频出现导致它成为热门，反应用户的主流兴趣，但是热门标签权重过大，不利于反应用户个性化的兴趣，因此我们借助TF-IDF算法的思想，过滤掉高频词（热门tag），保留有区分度的词语（tag），从而通过差异化实现个性化推荐

![image](https://user-images.githubusercontent.com/68730894/115217104-ba8ce080-a137-11eb-9934-c3f642d079e4.png)


如果一个tag很热门，会导致 user_tags[t] 很大，所以即使 tag_items[u,t] 很小，也会导致 score(u,i) 很大。给热门标签过大的权重，不能反应用户个性化的兴趣这里借鉴TF-IDF的思想，使用 tag_users[t] 表示标签t被多少个不同的用户使用。


### NormTagBased算法
对score进行归一化

![image](https://user-images.githubusercontent.com/68730894/115217130-c2e51b80-a137-11eb-8d95-63cd99406059.png)

```python 
user_item_mapping = pd.DataFrame(train_data.groupby(['userID', 'bookmarkID'])['bookmarkID'].count()).reset_index(inplace=True)  
#重新命名bookmarkID的统计列为total  
user_item_mapping.columns=['total']  
	  
user_tag_mapping = pd.DataFrame(self.train_data.groupby(['userID', 'tagID'])['tagID'].count()).reset_index(inplace=True)  
user_tag_mapping.columns=['total']  
  
items_tag_mapping = pd.DataFrame(self.train_data.groupby(['bookmarkID', 'tagID'])['tagID'].count()).reset_index(inplace=True)  
items_tag_mapping.columns=['total'] 
```

### 数据集
* 数据集：https://grouplens.org/datasets/hetrec-2011/
Delicious Bookmarks，105,000 bookmarks from 1867 users.

* 原始数据格式
![image](https://user-images.githubusercontent.com/68730894/115217241-e27c4400-a137-11eb-864c-feecf5c32c0e.png)

* 字典类型，保存了user对item的tag，即{userid: {item1:[tag1, tag2], ...}}
![image](https://user-images.githubusercontent.com/68730894/115217280-ec9e4280-a137-11eb-8e79-1c173230bf0d.png)

### 评价指标：召回率（Recall）和精确率（Precision）
使用精确率的目的是：在预测为正样本的结果中，我们有多少把握可以预测正确。算法是：在所有被预测为正的样本中实际为正的样本的概率。

准确率仅代表预测准确的程度。

使用召回率的目的是：它的含义类似：宁可错杀一千，绝不放过一个。

我们更关心top-N的出现，不能错放过任何一个top-N范围内的item。因为如果我们过多的将非top-N的商品放入top-N内，这样后续可能发生的非top-N内的item远超过top-N内的情况，造成严重偿失。召回率越高，代表实际top-N内被预测出来的概率越高。


```python 
def PrecisionAndRecall(recommend, top_N):
    hit = 0
    h_recall = 0
    h_precision = 0
    for user, items in test_data.item():
        if user not in train_data:
            continue
        # # 获取Top-N推荐列表  
	        rank = recommend(user, top_N)  
	        for item,rui in rank:  
	            if item in items:  
	                hit = hit + 1  
	        h_recall = h_recall + len(items)  
	        h_precision = h_precision + top_N  
    return (hit/(h_precision*1.0)), (hit/(h_recall*1.0))  
```

![算法比较](https://user-images.githubusercontent.com/68730894/115875049-68212c00-a477-11eb-9073-89b07a901b76.jpg)

<table class=MsoTableGrid border=1 cellspacing=0 cellpadding=0 width="100%"
 style='width:100.0%;border-collapse:collapse;border:none'>
 <tr style='height:14.0pt'>
  <td width="13%" nowrap rowspan=2 valign=top style='width:13.06%;border:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-size:11.0pt;color:black'>top-N</span></b></p>
  </td>
  <td width="28%" nowrap colspan=2 valign=top style='width:28.88%;border:solid windowtext 1.0pt;
  border-left:none;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-size:11.0pt;color:black'>SimpleTag-Based</span></b></p>
  </td>
  <td width="27%" nowrap colspan=2 valign=top style='width:27.26%;border:solid windowtext 1.0pt;
  border-left:none;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='color:black'>TFIDFTag-Based</span></b></p>
  </td>
  <td width="30%" nowrap colspan=2 valign=top style='width:30.8%;border:solid windowtext 1.0pt;
  border-left:none;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-size:11.0pt;color:black'>NormTag-Based</span></b></p>
  </td>
 </tr>
 <tr style='height:14.0pt'>
  <td width="14%" nowrap valign=top style='width:14.44%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  style='font-size:11.0pt;color:black'>精确率</span></b></p>
  </td>
  <td width="14%" nowrap valign=top style='width:14.44%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  style='font-size:11.0pt;color:black'>召回率</span></b></p>
  </td>
  <td width="13%" nowrap valign=top style='width:13.62%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  style='font-size:11.0pt;color:black'>精确率</span></b></p>
  </td>
  <td width="13%" nowrap valign=top style='width:13.62%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  style='font-size:11.0pt;color:black'>召回率</span></b></p>
  </td>
  <td width="15%" nowrap valign=top style='width:15.4%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  style='font-size:11.0pt;color:black'>精确率</span></b></p>
  </td>
  <td width="15%" nowrap valign=top style='width:15.4%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  style='font-size:11.0pt;color:black'>召回率</span></b></p>
  </td>
 </tr>
 <tr style='height:14.0pt'>
  <td width="13%" nowrap valign=top style='width:13.06%;border:solid windowtext 1.0pt;
  border-top:none;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='color:black'>5</span></p>
  </td>
  <td width="14%" nowrap valign=top style='width:14.44%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>0.829%</span></p>
  </td>
  <td width="14%" nowrap valign=top style='width:14.44%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>0.355%</span></p>
  </td>
  <td width="13%" nowrap valign=top style='width:13.62%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>1.008%</span></p>
  </td>
  <td width="13%" nowrap valign=top style='width:13.62%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>0.431%</span></p>
  </td>
  <td width="15%" nowrap valign=top style='width:15.4%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>1.153%</span></p>
  </td>
  <td width="15%" nowrap valign=top style='width:15.4%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>0.494%</span></p>
  </td>
 </tr>
 <tr style='height:14.0pt'>
  <td width="13%" nowrap valign=top style='width:13.06%;border:solid windowtext 1.0pt;
  border-top:none;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='color:black'>10</span></p>
  </td>
  <td width="14%" nowrap valign=top style='width:14.44%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>0.633%</span></p>
  </td>
  <td width="14%" nowrap valign=top style='width:14.44%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>0.542%</span></p>
  </td>
  <td width="13%" nowrap valign=top style='width:13.62%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>0.761%</span></p>
  </td>
  <td width="13%" nowrap valign=top style='width:13.62%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>0.652%</span></p>
  </td>
  <td width="15%" nowrap valign=top style='width:15.4%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>0.761%</span></p>
  </td>
  <td width="15%" nowrap valign=top style='width:15.4%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>0.652%</span></p>
  </td>
 </tr>
 <tr style='height:14.0pt'>
  <td width="13%" nowrap valign=top style='width:13.06%;border:solid windowtext 1.0pt;
  border-top:none;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='color:black'>20</span></p>
  </td>
  <td width="14%" nowrap valign=top style='width:14.44%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>0.512%</span></p>
  </td>
  <td width="14%" nowrap valign=top style='width:14.44%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>0.877%</span></p>
  </td>
  <td width="13%" nowrap valign=top style='width:13.62%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>0.549%</span></p>
  </td>
  <td width="13%" nowrap valign=top style='width:13.62%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>0.940%</span></p>
  </td>
  <td width="15%" nowrap valign=top style='width:15.4%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>0.568%</span></p>
  </td>
  <td width="15%" nowrap valign=top style='width:15.4%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>0.973%</span></p>
  </td>
 </tr>
 <tr style='height:14.0pt'>
  <td width="13%" nowrap valign=top style='width:13.06%;border:solid windowtext 1.0pt;
  border-top:none;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='color:black'>40</span></p>
  </td>
  <td width="14%" nowrap valign=top style='width:14.44%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>0.381%</span></p>
  </td>
  <td width="14%" nowrap valign=top style='width:14.44%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>1.304%</span></p>
  </td>
  <td width="13%" nowrap valign=top style='width:13.62%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>0.402%</span></p>
  </td>
  <td width="13%" nowrap valign=top style='width:13.62%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>1.376%</span></p>
  </td>
  <td width="15%" nowrap valign=top style='width:15.4%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>0.426%</span></p>
  </td>
  <td width="15%" nowrap valign=top style='width:15.4%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>1.457%</span></p>
  </td>
 </tr>
 <tr style='height:14.0pt'>
  <td width="13%" nowrap valign=top style='width:13.06%;border:solid windowtext 1.0pt;
  border-top:none;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='color:black'>60</span></p>
  </td>
  <td width="14%" nowrap valign=top style='width:14.44%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>0.318%</span></p>
  </td>
  <td width="14%" nowrap valign=top style='width:14.44%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>1.635%</span></p>
  </td>
  <td width="13%" nowrap valign=top style='width:13.62%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>0.328%</span></p>
  </td>
  <td width="13%" nowrap valign=top style='width:13.62%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>1.687%</span></p>
  </td>
  <td width="15%" nowrap valign=top style='width:15.4%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>0.356%</span></p>
  </td>
  <td width="15%" nowrap valign=top style='width:15.4%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>1.826%</span></p>
  </td>
 </tr>
 <tr style='height:14.0pt'>
  <td width="13%" nowrap valign=top style='width:13.06%;border:solid windowtext 1.0pt;
  border-top:none;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='color:black'>80</span></p>
  </td>
  <td width="14%" nowrap valign=top style='width:14.44%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>0.276%</span></p>
  </td>
  <td width="14%" nowrap valign=top style='width:14.44%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>1.893%</span></p>
  </td>
  <td width="13%" nowrap valign=top style='width:13.62%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>0.297%</span></p>
  </td>
  <td width="13%" nowrap valign=top style='width:13.62%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>2.033%</span></p>
  </td>
  <td width="15%" nowrap valign=top style='width:15.4%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>0.309%</span></p>
  </td>
  <td width="15%" nowrap valign=top style='width:15.4%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>2.119%</span></p>
  </td>
 </tr>
 <tr style='height:14.0pt'>
  <td width="13%" nowrap valign=top style='width:13.06%;border:solid windowtext 1.0pt;
  border-top:none;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='color:black'>100</span></p>
  </td>
  <td width="14%" nowrap valign=top style='width:14.44%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>0.248%</span></p>
  </td>
  <td width="14%" nowrap valign=top style='width:14.44%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>2.124%</span></p>
  </td>
  <td width="13%" nowrap valign=top style='width:13.62%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>0.269%</span></p>
  </td>
  <td width="13%" nowrap valign=top style='width:13.62%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>2.306%</span></p>
  </td>
  <td width="15%" nowrap valign=top style='width:15.4%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>0.277%</span></p>
  </td>
  <td width="15%" nowrap valign=top style='width:15.4%;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right'><span lang=EN-US
  style='font-size:11.0pt;color:black'>2.368%</span></p>
  </td>
 </tr>
</table>
