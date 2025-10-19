import pandas as pd

# 读取数据文件
df = pd.read_csv('customer_base.csv')
print("customer_base.csv的列名：")
print(list(df.columns))

# 查看前几行数据
print("ncustomer_base.csv前5行数据：")
print(df.head())