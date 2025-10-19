"""
分析客户聚类结果中不同cluster的特征分布,打印出来完整的特征均值，然后再做分析。
基于选择的特征，进行均值打印。
"""
import pandas as pd
import numpy as np

# 读取聚类结果文件
df = pd.read_csv('customer_clusters_result.csv')

# 定义用于聚类的特征（从customer_clustering.py中获取）
clustering_features = [
    # 收入与资产
    'monthly_income',
    'total_assets',
    'deposit_balance',
    'financial_balance',
    'fund_balance',
    'insurance_balance',
    
    # 消费与投资行为
    'credit_card_monthly_expense',
    'investment_monthly_count',
    
    # 产品持有情况
    'product_count',
    'financial_repurchase_count',  # 复购次数体现忠诚度
    
    # 线上活跃度
    'app_login_count',
    'app_financial_view_time',
    'app_product_compare_count',
    
    # 客户基础信息
    'age'
]

print("各cluster在聚类特征上的详细均值:")
print("="*60)

# 计算每个cluster的特征均值
cluster_feature_means = df.groupby('cluster')[clustering_features].mean()

# 为了便于比较，保留3位小数
cluster_feature_means = cluster_feature_means.round(3)

# 重新排序，按cluster编号排序
cluster_feature_means = cluster_feature_means.reindex(sorted(cluster_feature_means.index))

print(cluster_feature_means)

print("\n\n各cluster的客户数量统计:")
print("="*40)
cluster_counts = df['cluster'].value_counts().sort_index()
for cluster_id in cluster_counts.index:
    print(f"Cluster {cluster_id}: {cluster_counts[cluster_id]} 个客户")

print("\n\n各cluster的详细特征对比 (按特征分组):")
print("="*80)

# 按特征类型分组展示
print("\n1. 收入与资产特征:")
income_asset_features = ['monthly_income', 'total_assets', 'deposit_balance', 'financial_balance', 'fund_balance', 'insurance_balance']
print(cluster_feature_means[income_asset_features])

print("\n2. 消费与投资行为特征:")
consumption_behavior_features = ['credit_card_monthly_expense', 'investment_monthly_count']
print(cluster_feature_means[consumption_behavior_features])

print("\n3. 产品持有情况特征:")
product_features = ['product_count', 'financial_repurchase_count']
print(cluster_feature_means[product_features])

print("\n4. 线上活跃度特征:")
online_activity_features = ['app_login_count', 'app_financial_view_time', 'app_product_compare_count']
print(cluster_feature_means[online_activity_features])

print("\n5. 客户基础信息特征:")
basic_features = ['age']
print(cluster_feature_means[basic_features])