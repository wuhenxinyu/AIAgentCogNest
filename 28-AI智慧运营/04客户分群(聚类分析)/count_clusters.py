"""
统计客户聚类结果中cluster列的唯一值数量和分布
"""
import pandas as pd

# 读取聚类结果文件
df = pd.read_csv('customer_clusters_result.csv')

# 统计cluster列中的唯一值数量
unique_clusters = df['cluster'].nunique()
cluster_counts = df['cluster'].value_counts().sort_index()

print(f"总共有 {unique_clusters} 个cluster:")
print(cluster_counts)
print()

# 同时查看cluster_name列的分布
if 'cluster_name' in df.columns:
    unique_cluster_names = df['cluster_name'].nunique()
    cluster_name_counts = df['cluster_name'].value_counts().sort_index()
    
    print(f"总共有 {unique_cluster_names} 个cluster名称:")
    print(cluster_name_counts)