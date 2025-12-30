"""
Prophet 模型预测交通流量,预测未来210天的流量并将预测结果保存到csv文件中。
"""
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# 读取数据
train = pd.read_csv('09-时间序列AI大赛(Prophet)/02交通流量预测/train.csv')

# 将Datetime字段转换为日期时间格式
train['Datetime'] = pd.to_datetime(train['Datetime'], format='%d-%m-%Y %H:%M')

# 按天聚合乘客数量
# 以日期为单位分组，统计每天的总乘客数
train['Date'] = train['Datetime'].dt.date

daily_data = train.groupby('Date')['Count'].sum().reset_index()
daily_data.rename(columns={'Date': 'ds', 'Count': 'y'}, inplace=True)
daily_data['ds'] = pd.to_datetime(daily_data['ds'])

# 建立并拟合Prophet模型
model = Prophet(yearly_seasonality=True, seasonality_prior_scale=0.1)
model.fit(daily_data)

# 生成未来210天的日期数据
future = model.make_future_dataframe(periods=210)

# 进行预测
forecast = model.predict(future)
# 查看forecast都有哪些列，trend趋势、 yearly趋势、weekly趋势
print(forecast.columns)

# 只保留未来210天的预测结果，包含日期、预测值、预测区间上下界
future_forecast = forecast.tail(210)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# 保存预测结果到csv文件，编码为utf-8-sig防止中文乱码
future_forecast.to_csv('09-时间序列AI大赛(Prophet)/02交通流量预测/jetrail_210days_forecast.csv', index=False, encoding='utf-8-sig')

# 可视化预测结果
fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)

# 保存图表为图片文件
fig1.savefig('09-时间序列AI大赛(Prophet)/02交通流量预测/jetrail_prophet210_forecast_plot.png', dpi=300)
fig2.savefig('09-时间序列AI大赛(Prophet)/02交通流量预测/jetrail_prophet210_forecast_components.png', dpi=300)

plt.show()  # 显示所有matplotlib生成的图表窗口

# 说明：
# ds：日期
# yhat：预测的乘客数量
# yhat_lower/yhat_upper：预测区间上下界
# 运行脚本后将在当前目录下生成预测结果csv和两张图片。 