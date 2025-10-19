import pandas as pd

df_result = pd.read_csv('customer_clusters_result.csv')
print('聚类结果统计:')
print(df_result['cluster_name'].value_counts())
print()
print('各聚类关键指标平均值:')
print(df_result.groupby('cluster_name')[['monthly_income', 'total_assets', 'financial_repurchase_count', 'age', 'credit_card_monthly_expense']].mean())