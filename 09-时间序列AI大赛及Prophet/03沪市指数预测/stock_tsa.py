"""
 使用tsa对沪市指数进行分析：trend, seasonal, residual
 分析结果：
    • 趋势：沪市指数呈现上升趋势
    • 季节性：沪市指数在每个月的第15天有一个峰值
    • 残留：沪市指数的残留部分比较稳定
"""
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

# 数据加载
data = pd.read_csv('09-时间序列AI大赛(Prophet)/03沪市指数预测/shanghai_index_1990_12_19_to_2020_03_12.csv', usecols=['Timestamp', 'Price'])
data.Timestamp = pd.to_datetime(data.Timestamp)
data = data.set_index('Timestamp')
data['Price'] = data['Price'].apply(pd.to_numeric, errors='ignore')
# 进行线性插补缺漏值
data.Price.interpolate(inplace=True)
#  返回三个部分 trend（趋势），seasonal（季节性）和residual (残留)
result = sm.tsa.seasonal_decompose(data.Price, period=288)
result.plot()
plt.show()