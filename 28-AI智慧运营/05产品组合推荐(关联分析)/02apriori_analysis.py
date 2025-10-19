"""
使用关联分析（产品组合推荐）,比如场景：挖掘存款/理财/基金/保险的频繁组合模式。
关联分析：
1、Apriori算法
2、关联规则挖掘
"""
import pandas as pd
from itertools import combinations
from collections import defaultdict


def load_data(file_path):
    """
    加载数据并提取产品持有情况
    """
    df = pd.read_csv(file_path)
    
    # 选择产品持有标志列
    product_flags = ['deposit_flag', 'financial_flag', 'fund_flag', 'insurance_flag']
    
    # 创建产品名称映射
    product_map = {
        'deposit_flag': '存款',
        'financial_flag': '理财',
        'fund_flag': '基金',
        'insurance_flag': '保险'
    }
    
    return df, product_flags, product_map


def get_transactions(df, product_flags, product_map):
    """
    将数据转换为事务格式，用于Apriori算法
    """
    transactions = []
    
    for idx, row in df.iterrows():
        transaction = []
        for flag in product_flags:
            if row[flag] == 1:
                transaction.append(product_map[flag])
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
    # 加载数据
    file_path = '28-AI智慧运营/01百万客群经营(数据分析)/customer_behavior_assets.csv'
    df, product_flags, product_map = load_data(file_path)
    
    # 获取事务数据
    transactions = get_transactions(df, product_flags, product_map)
    
    print(f"总共有 {len(transactions)} 个事务")
    print(f"事务示例: {transactions[:5]}")
    
    # 设置最小支持度和置信度
    min_support = 0.01  # 可根据需要调整
    min_confidence = 0.1  # 可根据需要调整
    
    print(f"\\n使用最小支持度: {min_support}, 最小置信度: {min_confidence}")
    
    # 执行Apriori算法
    frequent_itemsets, num_transactions = apriori(transactions, min_support)
    
    print(f"\\n找到 {len(frequent_itemsets)} 个频繁项集")
    print("\\n频繁项集 (按支持度降序):")

    sorted_itemsets = sorted(frequent_itemsets.items(), key=lambda x: x[1], reverse=True)
    for itemset, support in sorted_itemsets:
        print(f"项集: {set(itemset)}, 支持度: {support:.4f}")
    
    # 生成关联规则
    rules = generate_rules(frequent_itemsets, transactions, min_confidence)
    
    print(f"\\n找到 {len(rules)} 条关联规则")
    print("\\n关联规则 (按置信度降序):")

    sorted_rules = sorted(rules, key=lambda x: x['confidence'], reverse=True)
    for rule in sorted_rules:
        print(f"{set(rule['antecedent'])} -> {set(rule['consequent'])}")
        print(f"  支持度: {rule['support']:.4f}, 置信度: {rule['confidence']:.4f}, 提升度: {rule['lift']:.4f}")
        print()


if __name__ == "__main__":
    main()