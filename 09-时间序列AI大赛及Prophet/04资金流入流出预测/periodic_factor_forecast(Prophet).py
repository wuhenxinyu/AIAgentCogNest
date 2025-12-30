"""
    周期因子分析与预测(使用Prophet模型)
    本脚本通过统计分析和可视化展示2014年3月至2014年8月的用户资金流入流出数据，
    使用Facebook Prophet模型预测未来30天的资金流入流出趋势。
    Prophet能够自动处理星期几和月份的季节性因素，以及趋势变化。
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from prophet import Prophet  # 导入Prophet库

# 设置中文字体，确保图表中的中文能正常显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 读取数据函数
def load_data(file_path):
    """
    从CSV文件读取用户资金余额数据，并进行日期筛选
    
    参数:
        file_path: CSV文件路径
    返回:
        df: 筛选后的DataFrame，包含2014-03-01至2014-08-31的数据
    """
    # 读取CSV文件
    df = pd.read_csv(file_path)
    # 将report_date字段从字符串格式(YYYYMMDD)转换为日期格式
    df['report_date'] = pd.to_datetime(df['report_date'], format='%Y%m%d')
    # 筛选2014年3月至2014年8月的数据作为训练集
    df = df[(df['report_date'] >= '2014-03-01') & (df['report_date'] <= '2014-08-31')]
    return df

# 按日期聚合数据函数
def aggregate_data(df):
    """
    按日期聚合每日的总申购金额和总赎回金额
    
    参数:
        df: 原始用户资金余额数据
    返回:
        daily_data: 按日期聚合后的每日资金流入流出数据
    """
    # 按report_date分组，计算每日的总申购金额和总赎回金额
    daily_data = df.groupby('report_date').agg({
        'total_purchase_amt': 'sum',  # 聚合计算每日总申购金额
        'total_redeem_amt': 'sum'     # 聚合计算每日总赎回金额
    }).reset_index()
    return daily_data

# 使用Prophet模型训练和预测函数
def train_and_predict_prophet(daily_data, forecast_days=30):
    """
    使用Prophet模型训练并预测未来指定天数的资金流入流出
    
    参数:
        daily_data: 包含历史数据的DataFrame
        forecast_days: 要预测的天数
    返回:
        forecast_df: 包含预测结果的DataFrame
        models: 包含申购和赎回金额预测模型的字典
    
    Prophet模型特点:
    - 自动处理每日、每周、每年的季节性模式
    - 能够捕获趋势变化
    - 对缺失数据和异常值有良好的鲁棒性
    """
    # 存储两个模型(申购和赎回)
    models = {}
    # 存储预测结果
    forecasts = {}
    
    # 为申购和赎回分别创建并训练Prophet模型
    for target in ['total_purchase_amt', 'total_redeem_amt']:
        # Prophet要求输入列名为'ds'(日期)和'y'(目标值)
        prophet_data = pd.DataFrame({
            'ds': daily_data['report_date'],
            'y': daily_data[target]
        })
        
        # 创建Prophet模型
        # yearly_seasonality设为False，因为数据只有6个月，不足以学习年度季节性
        # weekly_seasonality设为True，学习星期几的季节性
        # daily_seasonality设为False，因为我们的日期粒度是天
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'  # 使用乘法模型，更适合具有增长趋势的数据
        )
        
        # 添加月度季节性因素
        # 由于Prophet默认不包含月度季节性，我们需要手动添加
        model.add_seasonality(
            name='monthly',
            period=30.5,  # 平均月度周期
            fourier_order=5  # 傅里叶级数阶数，控制季节性的灵活性
        )
        
        # 训练模型
        model.fit(prophet_data)
        models[target] = model
        
        # 创建未来日期的数据框
        future = model.make_future_dataframe(periods=forecast_days)
        
        # 预测
        forecast = model.predict(future)
        
        # 只保留未来30天的预测结果
        future_forecast = forecast.tail(forecast_days)
        
        # 提取我们需要的预测值
        forecasts[target] = future_forecast
    
    # 合并申购和赎回的预测结果
    forecast_df = pd.DataFrame({
        'report_date': forecasts['total_purchase_amt']['ds'],
        'purchase': forecasts['total_purchase_amt']['yhat'],  # 预测的申购金额
        'redeem': forecasts['total_redeem_amt']['yhat']       # 预测的赎回金额
    })
    
    # 将日期格式转换为YYYYMMDD字符串格式
    forecast_df['report_date'] = forecast_df['report_date'].dt.strftime('%Y%m%d')
    
    return forecast_df, models

# 可视化结果函数
def visualize_results(daily_data, forecast_df, models):
    """
    可视化历史数据、预测结果和模型组件
    
    参数:
        daily_data: 历史数据
        forecast_df: 预测结果数据
        models: 包含申购和赎回金额预测模型的字典
    """
    # 创建一个包含多个子图的图表
    plt.figure(figsize=(20, 25))
    
    # 1. 绘制申购金额图表
    ax1 = plt.subplot(4, 2, 1)
    ax1.plot(pd.to_datetime(daily_data['report_date']), daily_data['total_purchase_amt'], label='历史申购金额')
    ax1.plot(pd.to_datetime(forecast_df['report_date']), forecast_df['purchase'], label='预测申购金额', linestyle='--')
    ax1.set_title('资金申购金额趋势及预测')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('金额')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 绘制赎回金额图表
    ax2 = plt.subplot(4, 2, 2)
    ax2.plot(pd.to_datetime(daily_data['report_date']), daily_data['total_redeem_amt'], label='历史赎回金额')
    ax2.plot(pd.to_datetime(forecast_df['report_date']), forecast_df['redeem'], label='预测赎回金额', linestyle='--')
    ax2.set_title('资金赎回金额趋势及预测')
    ax2.set_xlabel('日期')
    ax2.set_ylabel('金额')
    ax2.legend()
    ax2.grid(True)
    
    # 3. 使用Prophet的内置可视化功能展示申购模型的预测分解
    fig3 = models['total_purchase_amt'].plot_components(models['total_purchase_amt'].predict(
        models['total_purchase_amt'].make_future_dataframe(periods=30)))
    plt.suptitle('申购金额模型组件分析', y=1.02)
    
    # 4. 使用Prophet的内置可视化功能展示赎回模型的预测分解
    fig4 = models['total_redeem_amt'].plot_components(models['total_redeem_amt'].predict(
        models['total_redeem_amt'].make_future_dataframe(periods=30)))
    plt.suptitle('赎回金额模型组件分析', y=1.02)
    
    # 调整布局，避免标签重叠
    plt.tight_layout()
    # 保存图表为PNG文件
    plt.savefig('periodic_factor_forecast_prophet.png')
    plt.close()  # 关闭图表，释放内存

# 主函数
def main():
    """
    主函数，整合数据读取、处理、建模和预测的完整流程
    """
    # 读取数据
    file_path = './09-时间序列AI大赛(prophet)/04资金流入流出预测/user_balance_table.csv'  # 使用相对路径
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        # 尝试使用绝对路径
        file_path = '/Users/clz/Downloads/xmkf/llm/AIAgentCogNest/09-时间序列AI大赛(prophet)/04资金流入流出预测/user_balance_table.csv'
        if not os.path.exists(file_path):
            print(f"错误：找不到数据文件 {file_path}")
            return
    
    df = load_data(file_path)
    
    # 按日期聚合数据
    daily_data = aggregate_data(df)
    
    # 使用Prophet模型训练并预测未来30天
    forecast_df, models = train_and_predict_prophet(daily_data, forecast_days=30)
    
    # 保存预测结果到CSV文件
    forecast_df.to_csv('periodic_factor_forecast_201409_prophet.csv', index=False)
    
    # 可视化结果
    visualize_results(daily_data, forecast_df, models)
    
    print(f"Prophet模型预测结果已保存到 periodic_factor_forecast_201409_prophet.csv")
    print(f"可视化结果已保存到 periodic_factor_forecast_prophet.png")

# 程序入口
if __name__ == '__main__':
    main()