"""
关联规则挖掘,关联分析 Apriori如何找到啤酒和尿布的组合,采用efficient_apriori工具包
1、使用mlxtend.frequent_patterns工具包效率较低，但返回参数较多。
2、使用efficient_apriori工具包效率较高，但返回参数较少。

关联规则的视角：
1、支持度：一个项集在所有交易中出现的次数占总交易次数的比例。
2、置信度：在所有出现A项集的交易中，出现B项集的次数占A项集出现次数的比例。
3、提升度：置信度与先验概率的比值，用于衡量关联规则的有效性。
4、不需要考虑用户一定时期内的偏好，而是基于Transaction。只要能将数据转换成Transaction，就可以做购物篮分析：
	Step1、把数据整理成id=>item形式，转换成transaction
	Step2、设定关联规则的参数（support、confident）挖掘关联规则
	Step3、按某个指标（lift、support等）对以关联规则排序

关联规则中的最小支持度、最小置信度该如何确定：
1、最小支持度：最小置信度是实验出来的，可以根据业务场景和数据特点来确定，一般取0.01-0.5之间。
   • 最小支持度：
		不同的数据集，最小值支持度差别较大。可能是0.01到0.5之间
		可以从高到低输出前20个项集的支持度作为参考
2、最小置信度：可以根据业务场景和数据特点来确定，一般取0.5-1之间。
   • 最小置信度：可能是0.5到1之间
   • 提升度：表示使用关联规则可以提升的倍数，是置信度与期望置信度的比值。提升度至少要大于1

apriori关联规则挖掘：
1、采用efficient_apriori工具包
2、采用mlxtend.frequent_patterns工具包
apriori优缺点：
1、efficient_apriori工具包：
	• 优点：效率高，返回参数较少。
	• 缺点：不支持直接计算提升度。
2、mlxtend.frequent_patterns工具包：
	• 优点：支持直接计算提升度，返回参数较多。
	• 缺点：效率较低。
"""
import pandas as pd
import time

# 数据加载
data = pd.read_csv('28-AI智慧运营/BreadBasket/BreadBasket_DMS.csv')
# 统一小写
data['Item'] = data['Item'].str.lower()
# 去掉none项
data = data.drop(data[data.Item == 'none'].index)

# 采用efficient_apriori工具包
def rule1():
	from efficient_apriori import apriori
	start = time.time()
	# 得到一维数组orders_series，并且将Transaction作为index, value为Item取值
	orders_series = data.set_index('Transaction')['Item']
	# 将数据集进行格式转换
	transactions = []
	temp_index = 0
	for i, v in orders_series.items():
		if i != temp_index:
			temp_set = set()
			temp_index = i
			temp_set.add(v)
			transactions.append(temp_set)
		else:
			temp_set.add(v)
	
	# 挖掘频繁项集和频繁规则
	itemsets, rules = apriori(transactions, min_support=0.02,  min_confidence=0.5)
	print('频繁项集：', itemsets)
	print('关联规则：', rules)
	end = time.time()
	print("用时：", end-start)

# 
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
# 采用mlxtend.frequent_patterns工具包
def rule2():
	from mlxtend.frequent_patterns import apriori
	from mlxtend.frequent_patterns import association_rules
	pd.options.display.max_columns=100
	start = time.time()
	# one hot编码 0/1
	hot_encoded_df=data.groupby(['Transaction','Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Transaction')
	hot_encoded_df = hot_encoded_df.applymap(encode_units)
	#print(hot_encoded_df)
	# Step1，先挖掘频繁项集
	frequent_itemsets = apriori(hot_encoded_df, min_support=0.02, use_colnames=True)
	# Step2，在频繁项集的基础上，计算关联规则
	rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
	print("频繁项集：", frequent_itemsets)
	# 输出满足 rules['lift'] >=1 并且 rules['confidence'] >=0.5 的关联规则
	print("关联规则：", rules[ (rules['lift'] >= 1) & (rules['confidence'] >= 0.5) ])
	#print(rules['confidence'])
	end = time.time()
	print("用时：", end-start)

#rule1()
#print('-'*100)
rule2()

#ml = machine learning