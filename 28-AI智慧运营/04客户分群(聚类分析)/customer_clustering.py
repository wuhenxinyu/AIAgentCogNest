"""
客户聚类分析

什么是聚类分析？
聚类分析是一种无监督学习方法，用于将数据集中的对象分组，使得同一组内的对象彼此相似，而不同组之间的对象差异较大。
详细说明：
1. 聚类分析的目标是将数据集中的对象分组，使得同一组内的对象彼此相似，而不同组之间的对象差异较大。
2. 聚类分析的方法包括K均值聚类、DBSCAN聚类等。
3. 聚类分析的结果是一个聚类标签，用于将数据集中的对象分配到不同的聚类中。
4. 聚类分析的评估指标包括轮廓系数、Davies-Bouldin索引等。
5. 轮廓系数：用于评估聚类结果的质量，取值范围[-1, 1]，值越大表示聚类结果越合理。
6. Davies-Bouldin索引：用于评估聚类结果的质量，取值范围[0, ∞)，值越小表示聚类结果越合理。
7. 聚类分析的结果需要根据业务场景和业务需求进行解释和应用。
8. 聚类分析的结果可以用于客户分群、客户行为分析、客户价值评估等业务场景。聚类是从无到有分群的过程，本身不存在是聚合出来的分组。
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')  # 设置后端，避免GUI问题
import matplotlib.pyplot as plt
# 设置matplotlib支持中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 1. 读取数据
df_base = pd.read_csv('28-AI智慧运营/04客户分群(聚类分析)/customer_base.csv')
df_assets = pd.read_csv('28-AI智慧运营/04客户分群(聚类分析)/customer_behavior_assets.csv')

print("数据加载完成...")
print(f"基础信息表形状: {df_base.shape}")
print(f"资产行为表形状: {df_assets.shape}")

# 2. 数据合并
df = pd.merge(df_base, df_assets, on='customer_id', how='inner')
print(f"合并后数据形状: {df.shape}")

# 3. 特征工程 - 选择用于聚类的关键特征
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

# 选取聚类特征
df_cluster = df[clustering_features].copy()

# 4. 数据清洗
print("\n数据清洗...")
print("缺失值情况:")
print(df_cluster.isnull().sum())

# 用中位数填充数值型缺失值
for col in df_cluster.columns:
    if df_cluster[col].isnull().sum() > 0:
        df_cluster[col].fillna(df_cluster[col].median(), inplace=True)

# 5. 异常值处理（使用IQR方法）
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = df[column].median()
    return df

# 对数值列处理异常值
for col in df_cluster.columns:
    df_cluster = remove_outliers(df_cluster, col)

# 6. 数据标准化
scaler = StandardScaler()
df_cluster_scaled = scaler.fit_transform(df_cluster)
df_cluster_scaled = pd.DataFrame(df_cluster_scaled, columns=clustering_features)

print("\n数据预处理完成！")
print(f"用于聚类的特征数: {len(clustering_features)}")
print(f"聚类数据形状: {df_cluster_scaled.shape}")

# 7. 实现聚类算法
print("\n开始聚类分析...")

# 使用肘部法则确定最佳聚类数
inertias = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_cluster_scaled)
    inertias.append(kmeans.inertia_)

# 绘制肘部图
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('聚类数量 K')
plt.ylabel('簇内误差平方和 (Inertia)')
plt.title('肘部法则确定最佳聚类数')
plt.grid(True)
plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
plt.close()  # 关闭图形以释放内存

# 执行K-means聚类 (使用K=4，可根据肘部图调整)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(df_cluster_scaled)

# 计算轮廓系数
sil_score = silhouette_score(df_cluster_scaled, cluster_labels)
print(f"K-means聚类轮廓系数: {sil_score:.3f}")

# 将聚类结果添加到原数据
df['cluster'] = cluster_labels

# DBSCAN聚类（作为对比）
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(df_cluster_scaled)

# 计算DBSCAN轮廓系数
if len(set(dbscan_labels)) > 1:  # 确保有多个聚类才计算轮廓系数
    dbscan_sil_score = silhouette_score(df_cluster_scaled, dbscan_labels)
    print(f"DBSCAN聚类轮廓系数: {dbscan_sil_score:.3f}")
else:
    print("DBSCAN未生成多个聚类，无法计算轮廓系数")

# 8. 主要使用K-means结果进行分析
print(f"\n使用K-means进行聚类，聚类数: {optimal_k}")
print("聚类分布情况:")
print(pd.Series(cluster_labels).value_counts().sort_index())

# 9. 分析各聚类的特征并命名
print("\n各聚类特征分析:")
cluster_analysis = df.groupby('cluster')[clustering_features].mean()
print(cluster_analysis)

# 为每个聚类命名
cluster_names = {}
for cluster_id in sorted(df['cluster'].unique()):
    cluster_data = df[df['cluster'] == cluster_id]
    
    # 计算该聚类在各个关键指标上的平均值
    avg_income = cluster_data['monthly_income'].mean()
    avg_assets = cluster_data['total_assets'].mean()
    avg_repurchase = cluster_data['financial_repurchase_count'].mean()
    avg_age = cluster_data['age'].mean()
    avg_expense = cluster_data['credit_card_monthly_expense'].mean()
    
    # 根据特征命名聚类
    if avg_repurchase > cluster_analysis['financial_repurchase_count'].mean() and avg_assets > cluster_analysis['total_assets'].mean():
        cluster_names[cluster_id] = '高复购高价值客户'
    elif avg_income > cluster_analysis['monthly_income'].mean() and avg_age > 35:
        cluster_names[cluster_id] = '中产家庭客户'
    elif avg_expense > cluster_analysis['credit_card_monthly_expense'].mean() and avg_age < 35:
        cluster_names[cluster_id] = '年轻高消费客户'
    else:
        cluster_names[cluster_id] = '普通价值客户'
    
    print(f"\n聚类 {cluster_id} ({cluster_names[cluster_id]}):")
    print(f"  平均月收入: {avg_income:.2f}")
    print(f"  平均总资产: {avg_assets:.2f}")
    print(f"  平均复购次数: {avg_repurchase:.2f}")
    print(f"  平均年龄: {avg_age:.2f}")
    print(f"  平均信用卡月消费: {avg_expense:.2f}")
    print(f"  客户数量: {len(cluster_data)}")

# 将聚类名称添加到数据框
df['cluster_name'] = df['cluster'].map(cluster_names)

print("\n聚类命名完成！")
print(df['cluster_name'].value_counts())

# 10. 生成可视化图表
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']  # 支持中文字体
plt.rcParams['axes.unicode_minus'] = False

# 聚类分布饼图
plt.figure(figsize=(10, 8))
cluster_counts = df['cluster_name'].value_counts()
plt.pie(cluster_counts.values, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('客户聚类分布图')
plt.savefig('customer_clusters_pie.png', dpi=300, bbox_inches='tight')
plt.close()  # 关闭图形以释放内存

# 聚类在主要特征上的对比（柱状图）
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('各客户群组特征对比', fontsize=16)

# 平均月收入对比
cluster_avg_income = df.groupby('cluster_name')['monthly_income'].mean().reindex(cluster_names.values())
axes[0, 0].bar(cluster_avg_income.index, cluster_avg_income.values)
axes[0, 0].set_title('各群组平均月收入')
axes[0, 0].tick_params(axis='x', rotation=45)

# 平均总资产对比
cluster_avg_assets = df.groupby('cluster_name')['total_assets'].mean().reindex(cluster_names.values())
axes[0, 1].bar(cluster_avg_assets.index, cluster_avg_assets.values)
axes[0, 1].set_title('各群组平均总资产')
axes[0, 1].tick_params(axis='x', rotation=45)

# 平均复购次数对比
cluster_avg_repurchase = df.groupby('cluster_name')['financial_repurchase_count'].mean().reindex(cluster_names.values())
axes[1, 0].bar(cluster_avg_repurchase.index, cluster_avg_repurchase.values)
axes[1, 0].set_title('各群组平均复购次数')
axes[1, 0].tick_params(axis='x', rotation=45)

# 平均年龄对比
cluster_avg_age = df.groupby('cluster_name')['age'].mean().reindex(cluster_names.values())
axes[1, 1].bar(cluster_avg_age.index, cluster_avg_age.values)
axes[1, 1].set_title('各群组平均年龄')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('cluster_features_comparison.png', dpi=300, bbox_inches='tight')
plt.close()  # 关闭图形以释放内存

# 使用PCA降维进行聚类结果可视化
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_cluster_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['cluster'], cmap='viridis', alpha=0.6)
plt.xlabel(f'第一主成分 (解释方差比: {pca.explained_variance_ratio_[0]:.2f})')
plt.ylabel(f'第二主成分 (解释方差比: {pca.explained_variance_ratio_[1]:.2f})')
plt.title('客户聚类PCA可视化')
plt.colorbar(scatter)

# 添加聚类中心
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='x', s=200, linewidths=3, label='聚类中心')
plt.legend()
plt.savefig('cluster_pca_visualization.png', dpi=300, bbox_inches='tight')
plt.close()  # 关闭图形以释放内存

# 保存聚类结果到CSV文件
df.to_csv('customer_clusters_result.csv', index=False, encoding='utf-8-sig')
print(f"\n聚类结果已保存到 'customer_clusters_result.csv'")
print(f"共生成了 {len(df)} 条客户记录的聚类标签")

print("\n客户聚类分析完成！")
print("生成的文件:")
print("- elbow_method.png: 肘部法则图")
print("- customer_clusters_pie.png: 聚类分布饼图")
print("- cluster_features_comparison.png: 特征对比图")
print("- cluster_pca_visualization.png: PCA可视化图")
print("- customer_clusters_result.csv: 包含聚类标签的完整数据")