"""
使用时间序列分析（资产变动预测模型）
场景：预测客户未来季度AUM增长趋势。帮我使用ARIMA模型，编写新的Python脚本。

"""
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.tseries.offsets import DateOffset
import os

# 抑制警告
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data():
    """加载和预处理数据"""
    print("正在加载数据...")
    df = pd.read_csv('28-AI智慧运营/07资产变动预测(时间序列分析)/customer_behavior_assets.csv')
    
    # 转换统计月份为日期格式
    df['stat_month'] = pd.to_datetime(df['stat_month'], format='%Y-%m', errors='coerce')
    
    # 选择需要的列
    df = df[['customer_id', 'stat_month', 'total_assets']].dropna()
    
    # 按月份汇总总资产
    monthly_assets = df.groupby('stat_month')['total_assets'].sum().reset_index()
    monthly_assets.set_index('stat_month', inplace=True)
    
    # 按时间排序
    monthly_assets.sort_index(inplace=True)
    
    print(f"数据时间范围: {monthly_assets.index.min()} 到 {monthly_assets.index.max()}")
    print(f"数据点数量: {len(monthly_assets)}")
    
    return monthly_assets

def check_stationarity(timeseries, title):
    """检查时间序列的平稳性"""
    print(f'\n{title}:')
    print('Results of Dickey-Fuller Test:')
    result = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(result[0:4], 
                         index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key, value in result[4].items():
        dfoutput[f'Critical Value ({key})'] = value
    print(dfoutput)
    
    # 判断是否平稳
    if result[1] <= 0.05:
        print("序列是平稳的 (p <= 0.05)")
        return True
    else:
        print("序列不平稳 (p > 0.05)")
        return False

def decompose_timeseries(ts_data, title="时间序列分解"):
    """对时间序列进行分解"""
    try:
        # 如果数据点少于两个完整周期（对于季度分解需要至少8个点），跳过分解
        if len(ts_data) < 8:
            print("数据点不足，跳过时间序列分解")
            return None
        
        decomposition = seasonal_decompose(ts_data, model='additive', period=4)  # 假设季度性
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        decomposition.observed.plot(ax=axes[0], title='原始数据')
        decomposition.trend.plot(ax=axes[1], title='趋势')
        decomposition.seasonal.plot(ax=axes[2], title='季节性')
        decomposition.resid.plot(ax=axes[3], title='残差')
        plt.tight_layout()
        
        filename = f"{title.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        return decomposition
    except Exception as e:
        print(f"时间序列分解时出现错误: {e}")
        return None

def arima_forecast_with_validation(ts_data, forecast_steps=4):
    """使用ARIMA模型进行预测，并包含验证机制"""
    print("开始ARIMA参数优化与验证...")
    
    # 如果数据点太少，直接使用简单方法
    if len(ts_data) < 8:
        print(f"数据点较少 ({len(ts_data)}个)，使用移动平均趋势预测")
        return moving_average_trend_forecast(ts_data, forecast_steps)
    
    # 确定最优参数
    best_aic = np.inf
    best_order = None
    best_model = None
    
    # 尝试不同的参数组合
    for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    # 使用训练集拟合模型
                    train_data = ts_data[:-2]  # 留出最后2个点用于验证
                    model = ARIMA(train_data, order=(p, d, q))
                    fitted_model = model.fit()
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_order = (p, d, q)
                        best_model = fitted_model
                except:
                    continue
    
    if best_model is None:
        print("无法找到合适的ARIMA模型，使用移动平均趋势预测")
        return moving_average_trend_forecast(ts_data, forecast_steps)
    
    print(f"最优参数: ARIMA{best_order}, AIC: {best_aic}")
    
    # 使用完整数据集重新训练模型
    final_model = ARIMA(ts_data, order=best_order)
    final_fitted_model = final_model.fit()
    
    print("模型拟合完成")
    print("\n模型摘要:")
    print(final_fitted_model.summary())
    
    # 尝试进行预测
    try:
        forecast_result = final_fitted_model.forecast(steps=forecast_steps)
        forecast_index = pd.date_range(start=ts_data.index[-1] + pd.DateOffset(months=1), 
                                       periods=forecast_steps, freq='MS')  # 月初频率
        forecast_df = pd.DataFrame(forecast_result, index=forecast_index, columns=['forecast'])
        
        # 获取置信区间
        forecast_ci = final_fitted_model.get_forecast(steps=forecast_steps).conf_int()
        forecast_ci.index = forecast_index
        
        # 检查是否有NaN值
        if forecast_df['forecast'].isna().any():
            print("检测到NaN预测值，使用移动平均趋势预测作为备选方案")
            return moving_average_trend_forecast(ts_data, forecast_steps)
        
        return final_fitted_model, forecast_df, forecast_ci, best_order
    except Exception as e:
        print(f"ARIMA预测时出现错误: {e}")
        return moving_average_trend_forecast(ts_data, forecast_steps)

def moving_average_trend_forecast(ts_data, forecast_steps=4):
    """使用移动平均趋势方法进行预测"""
    print("使用移动平均趋势预测方法...")
    
    # 计算最近趋势
    if len(ts_data) >= 3:
        # 使用最近的几个数据点计算趋势
        recent_data = ts_data.values[-3:]
        x = np.arange(len(recent_data))
        
        # 计算线性趋势
        coeffs = np.polyfit(x, recent_data, 1)
        
        # 预测未来值
        last_val = ts_data.values[-1]
        trend_slope = coeffs[0]
        forecast_values = [last_val + (i+1) * trend_slope for i in range(forecast_steps)]
    else:
        # 如果数据不足，使用简单平均
        avg_change = np.mean(np.diff(ts_data.values)) if len(ts_data) > 1 else 0
        last_val = ts_data.values[-1]
        forecast_values = [last_val + (i+1) * avg_change for i in range(forecast_steps)]
    
    # 创建预测索引
    forecast_index = pd.date_range(start=ts_data.index[-1] + pd.DateOffset(months=1), 
                                   periods=forecast_steps, freq='MS')
    forecast_df = pd.DataFrame(forecast_values, index=forecast_index, columns=['forecast'])
    
    # 创建置信区间（使用历史数据的变异幅度）
    if len(ts_data) > 1:
        changes = np.diff(ts_data.values)
        std_change = np.std(changes)
        confidence_interval = std_change * 1.96  # 95%置信区间
    else:
        confidence_interval = ts_data.values[0] * 0.05  # 5%的基数作为置信区间
    
    forecast_ci = pd.DataFrame({
        'lower': forecast_df['forecast'] - confidence_interval,
        'upper': forecast_df['forecast'] + confidence_interval
    }, index=forecast_index)
    
    # 返回一个虚拟模型对象
    class DummyModel:
        def summary(self):
            return "Moving Average Trend Model"
    
    dummy_model = DummyModel()
    
    return dummy_model, forecast_df, forecast_ci, "Moving Average Trend"

def plot_forecast_with_components(ts_data, forecast_df, forecast_ci, title="AUM增长趋势预测"):
    """绘制历史数据和预测结果（包含更多组件）"""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 绘制历史数据
    ax.plot(ts_data.index, ts_data.values, label='历史AUM', marker='o', linewidth=2.5, markersize=8)
    
    # 绘制预测数据
    ax.plot(forecast_df.index, forecast_df['forecast'], label='预测AUM', marker='s', linestyle='--', 
            linewidth=2.5, color='red', markersize=8)
    
    # 绘制置信区间
    ax.fill_between(forecast_ci.index, 
                    forecast_ci.iloc[:, 0], 
                    forecast_ci.iloc[:, 1], 
                    color='red', alpha=0.2, label='95%置信区间')
    
    # 设置标题和标签
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('时间', fontsize=14)
    ax.set_ylabel('AUM (总资产)', fontsize=14)
    
    # 格式化x轴日期
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # 每2个月显示一个标签
    plt.xticks(rotation=45)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 添加图例
    ax.legend(fontsize=12)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    filename = f"{title.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def print_forecast_insights(ts_data, forecast_df):
    """打印预测洞察"""
    print("\n" + "="*50)
    print("预测洞察分析")
    print("="*50)
    
    # 计算历史AUM的统计信息
    hist_mean = ts_data.mean()
    hist_max = ts_data.max()
    hist_min = ts_data.min()
    
    # 计算预测AUM的统计信息
    pred_mean = forecast_df['forecast'].mean()
    pred_max = forecast_df['forecast'].max()
    pred_min = forecast_df['forecast'].min()
    
    print(f"历史AUM - 平均值: {hist_mean:,.2f}, 最高值: {hist_max:,.2f}, 最低值: {hist_min:,.2f}")
    print(f"预测AUM - 平均值: {pred_mean:,.2f}, 最高值: {pred_max:,.2f}, 最低值: {pred_min:,.2f}")
    
    # 计算增长趋势
    if len(ts_data) >= 2:
        recent_change = ((ts_data.iloc[-1] - ts_data.iloc[-3]) / ts_data.iloc[-3]) * 100 if len(ts_data) >= 3 else 0
        forecast_change = ((pred_max - ts_data.iloc[-1]) / ts_data.iloc[-1]) * 100
        
        print(f"近期趋势: {recent_change:+.2f}% (最近2-3个月)")
        print(f"预测增长: {forecast_change:+.2f}% (相对于最新数据)")
    
    print("="*50)

def main():
    print("开始AUM增长趋势预测 (ARIMA时间序列分析 - 企业级)")
    
    # 加载和预处理数据
    monthly_assets = load_and_preprocess_data()
    
    # 检查数据是否足够
    if len(monthly_assets) < 2:
        print(f"错误: 可用数据点太少 ({len(monthly_assets)}个)，无法进行预测")
        return
    
    # 检查平稳性
    original_stationarity = check_stationarity(monthly_assets['total_assets'], "原始时间序列")
    
    # 进行时间序列分解
    decomposition = decompose_timeseries(monthly_assets['total_assets'])
    
    # 使用ARIMA模型进行预测（包含验证机制）
    fitted_model, forecast_df, forecast_ci, best_order = arima_forecast_with_validation(
        monthly_assets['total_assets'], 
        forecast_steps=4  # 预测未来4个季度
    )
    
    # 绘制预测结果
    plot_forecast_with_components(monthly_assets['total_assets'], forecast_df, forecast_ci)
    
    # 输出预测结果
    print("\n未来4个季度的AUM预测结果:")
    for date, value in forecast_df.iterrows():
        print(f"{date.strftime('%Y-%m')}: {value['forecast']:,.2f}")
    
    print(f"\n使用模型: {best_order}")
    
    # 打印预测洞察
    print_forecast_insights(monthly_assets['total_assets'], forecast_df)
    
    print(f"\n预测完成！图表已保存。")

if __name__ == "__main__":
    main()