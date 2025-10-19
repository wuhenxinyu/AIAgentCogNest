import pandas as pd

# 读取数据
df_base = pd.read_csv('customer_base.csv')
df_assets = pd.read_csv('customer_behavior_assets.csv')

print('customer_base.csv 列名:')
print(df_base.columns.tolist())
print()
print('customer_behavior_assets.csv 列名:')
print(df_assets.columns.tolist())