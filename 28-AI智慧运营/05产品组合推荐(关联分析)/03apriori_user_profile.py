"""
用户画像分析：
1、加载客户基础信息和行为资产数据
2、将用户属性分类为离散值
3、将客户行为资产分类为离散值
"""
import pandas as pd
from itertools import combinations
from collections import defaultdict
import numpy as np


def load_data():
    """
    加载客户基础信息和行为资产数据
    """
    # 加载客户基础信息
    base_df = pd.read_csv('28-AI智慧运营/01百万客群经营(数据分析)/customer_base.csv')
    
    # 加载客户行为资产信息
    behavior_df = pd.read_csv('28-AI智慧运营/01百万客群经营(数据分析)/customer_behavior_assets.csv')
    
    return base_df, behavior_df


def categorize_user_attributes(row):
    """
    将用户属性分类为离散值
    """
    attributes = []
    
    # 年龄分组
    age = row['age']
    if 18 <= age < 30:
        attributes.append('年龄_18-30')
    elif 30 <= age < 40:
        attributes.append('年龄_30-40')
    elif 40 <= age < 50:
        attributes.append('年龄_40-50')
    elif 50 <= age < 60:
        attributes.append('年龄_50-60')
    else:
        attributes.append('年龄_60+')
    
    # 性别
    gender = row['gender']
    attributes.append(f'性别_{gender}')
    
    # 职业类型
    occupation_type = row['occupation_type']
    if pd.notna(occupation_type):
        attributes.append(f'职业类型_{occupation_type}')
    
    # 月收入分组
    income = row['monthly_income']
    if income < 10000:
        attributes.append('收入_<1万')
    elif 10000 <= income < 30000:
        attributes.append('收入_1-3万')
    elif 30000 <= income < 50000:
        attributes.append('收入_3-5万')
    elif 50000 <= income < 80000:
        attributes.append('收入_5-8万')
    else:
        attributes.append('收入_>=8万')
    
    # 生命周期阶段
    lifecycle = row['lifecycle_stage']
    if pd.notna(lifecycle):
        attributes.append(f'生命周期_{lifecycle}')
    
    # 婚姻状况
    marriage = row['marriage_status']
    if pd.notna(marriage):
        attributes.append(f'婚姻_{marriage}')
    
    # 城市级别
    city = row['city_level']
    if pd.notna(city):
        attributes.append(f'城市_{city}')
    
    return attributes


def categorize_products(row):
    """
    将产品持有情况分类
    """
    products = []
    
    # 产品持有标志
    product_flags = [
        ('deposit_flag', '存款'),
        ('financial_flag', '理财'),
        ('fund_flag', '基金'),
        ('insurance_flag', '保险')
    ]
    
    for flag, product_name in product_flags:
        if row[flag] == 1:
            products.append(product_name)
    
    return products


def create_transactions(base_df, behavior_df):
    """
    将客户基础信息和产品持有情况合并为事务
    """
    # 按customer_id合并数据
    merged_df = pd.merge(base_df, behavior_df, on='customer_id', how='inner')
    
    transactions = []
    
    for idx, row in merged_df.iterrows():
        # 获取用户画像属性
        user_attributes = categorize_user_attributes(row)
        
        # 获取产品持有情况
        products = categorize_products(row)
        
        # 合并用户属性和产品
        transaction = user_attributes + products
        
        if transaction:  # 只添加非空事务
            transactions.append(transaction)
    
    return transactions


def get_itemsets(transactions, min_support):
    """
    获取满足最小支持度的1-项集
    """
    item_counts = defaultdict(int)
    num_transactions = len(transactions)
    
    # 统计单个项的出现次数
    for transaction in transactions:
        for item in transaction:
            item_counts[frozenset([item])] += 1
    
    # 计算支持度并过滤
    itemsets = {}
    for itemset, count in item_counts.items():
        support = count / num_transactions
        if support >= min_support:
            itemsets[itemset] = support
    
    return itemsets, num_transactions


def join_set(itemset, length):
    """
    通过连接操作生成候选k-项集
    """
    return set([item1.union(item2) for item1 in itemset for item2 in itemset 
                if len(item1.union(item2)) == length])


def prune_set(candidate_set, transactions, min_support, num_transactions):
    """
    剪枝操作，过滤不满足最小支持度的候选项集
    """
    item_counts = defaultdict(int)
    
    for transaction in transactions:
        for candidate in candidate_set:
            if candidate.issubset(set(transaction)):
                item_counts[candidate] += 1
    
    # 计算支持度并过滤
    pruned_set = {}
    for itemset, count in item_counts.items():
        support = count / num_transactions
        if support >= min_support:
            pruned_set[itemset] = support
    
    return pruned_set


def apriori(transactions, min_support=0.1):
    """
    Apriori算法实现
    """
    # 获取1-项集
    L1, num_transactions = get_itemsets(transactions, min_support)
    L_set = [L1]
    k = 2
    
    while L_set[k-2]:
        # 连接操作生成候选k-项集
        Ck = join_set(L_set[k-2].keys(), k)
        
        # 剪枝操作
        Lk = prune_set(Ck, transactions, min_support, num_transactions)
        
        L_set.append(Lk)
        k += 1
    
    # 合并所有频繁项集
    all_frequent_itemsets = {}
    for itemsets in L_set:
        all_frequent_itemsets.update(itemsets)
    
    return all_frequent_itemsets, num_transactions


def generate_rules(frequent_itemsets, transactions, min_confidence=0.5):
    """
    生成关联规则
    """
    rules = []
    
    for itemset, support in frequent_itemsets.items():
        if len(itemset) > 1:
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    
                    # 计算置信度
                    antecedent_support = sum(1 for transaction in transactions 
                                             if antecedent.issubset(set(transaction))) / len(transactions)
                    
                    if antecedent_support > 0:
                        confidence = support / antecedent_support
                        
                        if confidence >= min_confidence:
                            rules.append({
                                'antecedent': antecedent,
                                'consequent': consequent,
                                'support': support,
                                'confidence': confidence,
                                'lift': confidence / (sum(1 for transaction in transactions 
                                                         if consequent.issubset(set(transaction))) / len(transactions))
                            })
    
    return rules


def main():
    print("开始加载数据...")
    base_df, behavior_df = load_data()
    
    print(f"客户基础信息数据: {len(base_df)} 条记录")
    print(f"客户行为资产数据: {len(behavior_df)} 条记录")
    
    print("\n开始创建事务...")
    transactions = create_transactions(base_df, behavior_df)
    
    print(f"总共有 {len(transactions)} 个事务")
    print(f"事务示例: {transactions[0] if transactions else []}")
    
    # 设置最小支持度和置信度
    min_support = 0.01  # 可根据需要调整
    min_confidence = 0.1  # 可根据需要调整
    
    print(f"\n使用最小支持度: {min_support}, 最小置信度: {min_confidence}")
    
    # 执行Apriori算法
    frequent_itemsets, num_transactions = apriori(transactions, min_support)
    
    print(f"\n找到 {len(frequent_itemsets)} 个频繁项集")
    print("\n前20个频繁项集 (按支持度降序):")
    sorted_itemsets = sorted(frequent_itemsets.items(), key=lambda x: x[1], reverse=True)
    for itemset, support in sorted_itemsets[:20]:
        print(f"项集: {set(itemset)}, 支持度: {support:.4f}")
    
    # 生成关联规则
    rules = generate_rules(frequent_itemsets, transactions, min_confidence)
    
    print(f"\n找到 {len(rules)} 条关联规则")
    print("\n前20条关联规则 (按置信度降序):")
    sorted_rules = sorted(rules, key=lambda x: x['confidence'], reverse=True)
    for i, rule in enumerate(sorted_rules[:20]):
        print(f"{set(rule['antecedent'])} -> {set(rule['consequent'])}")
        print(f"  支持度: {rule['support']:.4f}, 置信度: {rule['confidence']:.4f}, 提升度: {rule['lift']:.4f}")
        print()


if __name__ == "__main__":
    main()