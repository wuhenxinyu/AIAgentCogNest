"""
用于交通流量预测场景，作用是对交通流量数据进行处理并使用Prophet模型进行未来流量预测。
首先，代码使用pandas库加载存储在'train.csv'文件中的数据，接着对日期列进行格式转换并设置为索引，移除无关列。
然后将数据按天重采样求和，转换为日级别数据。之后添加Prophet模型所需的'ds'和'y'列，并移除冗余列。
使用Prophet模型，开启年季节性并调整季节性先验尺度，拟合日级别数据后生成未来约7个月（213天）的日期数据框进行预测。
最后绘制预测结果图和各成分图，以便直观查看预测结果及趋势、季节性等成分。

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 数据加载
train = pd.read_csv('09-时间序列AI大赛(Prophet)/02交通流量预测/train.csv')
print(train.head())

# 转换日期列数据格式为pandas的日期格式，便于后续时间序列操作
train['Datetime'] = pd.to_datetime(train.Datetime, format='%d-%m-%Y %H:%M')
# 将Datetime列设置为索引，方便按时间进行数据处理
train.index = train.Datetime
# 移除ID和Datetime列，保留分析所需核心数据
train.drop(['ID', 'Datetime'], axis=1, inplace=True)

# 按天对数据进行重采样并求和，将数据聚合到日级别
daily_train = train.resample('D').sum()
print(daily_train)
# 添加ds列，该列为Prophet模型所需的时间列
daily_train['ds'] = daily_train.index
# 添加y列，该列为Prophet模型所需的目标值列
daily_train['y'] = daily_train.Count
# 移除原有的Count列，避免数据冗余
daily_train.drop(['Count'], axis=1, inplace=True)
print(daily_train)

from prophet import Prophet
# 初始化Prophet模型，设置年季节性为True，并调整季节性先验尺度
m = Prophet(yearly_seasonality=True, seasonality_prior_scale=0.1)
# 使用日级别数据拟合Prophet模型
m.fit(daily_train)
# 生成未来213天（约7个月）的日期数据框，用于预测
future = m.make_future_dataframe(periods=213)
# 对未来日期进行预测
forecast = m.predict(future)
# 查看forecast都有哪些列
print(forecast.columns)
# 绘制预测结果图
fig1 = m.plot(forecast)

# 绘制预测结果的各个成分图，查看趋势、季节性等成分
fig2 = m.plot_components(forecast)

# 保存图表为图片文件
fig1.savefig('09-时间序列AI大赛(Prophet)/02交通流量预测/jetrail_prophet_forecast_plot.png', dpi=300)
fig2.savefig('09-时间序列AI大赛(Prophet)/02交通流量预测/jetrail_prophet_forecast_components.png', dpi=300)

plt.show()  # 显示所有matplotlib生成的图表窗口