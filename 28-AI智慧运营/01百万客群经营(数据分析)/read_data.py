"""
帮我编写Python，读取 customer_base.csv 和customer_behavior_assets.csv 的前5行数据。
通过运行，可以让AI看到前5行的数据，方便理解数据表的字段含义
"""
import pandas as pd

# 读取 customer_base.csv 的前5行
try:
    df_base = pd.read_csv('customer_base.csv', encoding='utf-8')
except UnicodeDecodeError:
    df_base = pd.read_csv('customer_base.csv', encoding='gbk')
print("customer_base.csv 前5行：")
print(df_base.head(5))
print("\n")

# 读取 customer_behavior_assets.csv 的前5行
try:
    df_assets = pd.read_csv('customer_behavior_assets.csv', encoding='utf-8')
except UnicodeDecodeError:
    df_assets = pd.read_csv('customer_behavior_assets.csv', encoding='gbk')
print("customer_behavior_assets.csv 前5行：")
print(df_assets.head(5))