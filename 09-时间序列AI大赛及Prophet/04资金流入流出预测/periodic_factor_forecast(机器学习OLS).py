"""
    周期因子分析与预测(机器学习OLS最小二乘法线性回归)
    本脚本通过统计分析和可视化展示2014年3月至2014年8月的用户资金流入流出数据，
    并根据星期几和日期的周期因子，预测未来30天的资金流入流出趋势。
    本脚本使用最小二乘法线性回归模型进行预测，模型的输入特征包括星期几和日期的哑变量。

    prompt:
        我想用周期因子对 total_purchase_amt 和 total_redeem_amt进行计算，使用 weekday 和 day的周期因子。
        weekday 和  day之间的影响可能有叠加。需要考虑历史的数据是在这种因素叠加的基础上产生的。
        然后对 未来30天，即从20140901开始，输出 report_date, purchse, redeem （那天的申购和赎回的金额），保存到 .csv中
        帮我编写Python代码 （可以使用机器学习，进行OLS（最小二乘法线性回归））
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from datetime import datetime, timedelta
import os

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

# 提取特征函数
def extract_features(daily_data):
    """
    从日期中提取周期因子特征，包括星期几和日期的哑变量
    
    参数:
        daily_data: 按日期聚合后的数据
    返回:
        daily_data: 添加了特征列的数据集
        features: 用于建模的特征矩阵(哑变量表示的周期因子)
    
    周期因子说明:
    1. 星期几因子(weekday): 捕捉一周内不同日期对资金流动的影响模式
    2. 日期因子(day_of_month): 捕捉一个月内不同日期对资金流动的影响模式
    
    哑变量转换原理:
    - 将分类变量(如星期几0-6)转换为多个二元变量，每个变量代表一个类别
    - drop_first=True: 移除第一个类别以避免多重共线性问题
    """
    # 提取星期几特征(0=周一, 1=周二, ..., 6=周日)
    daily_data['weekday'] = daily_data['report_date'].dt.weekday
    # 提取日期中的几号特征(1-31)
    daily_data['day_of_month'] = daily_data['report_date'].dt.day
    
    # 创建星期几的哑变量矩阵
    # 例如: weekday=0(周一) 会生成 weekday_1=0, weekday_2=0, ..., weekday_6=0
    weekday_dummies = pd.get_dummies(daily_data['weekday'], prefix='weekday', drop_first=True)
    
    # 创建日期的哑变量矩阵
    # 例如: day_of_month=1 会生成 day_2=0, day_3=0, ..., day_31=0
    day_dummies = pd.get_dummies(daily_data['day_of_month'], prefix='day', drop_first=True)
    
    # 合并星期几和日期的哑变量特征
    features = pd.concat([weekday_dummies, day_dummies], axis=1)
    return daily_data, features

# 训练模型并预测函数
def train_and_predict(daily_data, features):
    """
    使用机器学习中的OLS(普通最小二乘法)线性回归模型，基于周期因子特征预测未来30天的资金流入流出
    
    参数:
        daily_data: 包含历史数据的DataFrame
        features: 特征矩阵(星期几和日期的哑变量)
    返回:
        forecast_df: 包含预测结果的DataFrame
        model_purchase: 申购金额预测模型
        model_redeem: 赎回金额预测模型
    
    OLS模型公式:
    申购金额 = β0 + β1*weekday_1 + β2*weekday_2 + ... + β30*day_31 + ε
    赎回金额 = γ0 + γ1*weekday_1 + γ2*weekday_2 + ... + γ30*day_31 + ζ
    其中: β0, γ0是截距项，βi, γi是各周期因子的系数，ε, ζ是误差项
    """
    # 准备预测的日期范围（2014-09-01至2014-09-30）
    future_dates = [datetime(2014, 9, i+1) for i in range(30)]
    future_df = pd.DataFrame({'report_date': future_dates})
    
    # 为预测数据提取特征
    future_df['weekday'] = future_df['report_date'].dt.weekday
    future_df['day_of_month'] = future_df['report_date'].dt.day
    
    # 创建预测数据的哑变量
    future_weekday_dummies = pd.get_dummies(future_df['weekday'], prefix='weekday', drop_first=True)
    future_day_dummies = pd.get_dummies(future_df['day_of_month'], prefix='day', drop_first=True)
    
    # 确保预测特征与训练特征列名一致
    future_features = pd.concat([future_weekday_dummies, future_day_dummies], axis=1)
    
    # 检查并添加缺失的列(在预测数据中可能不存在的哑变量列)
    for col in features.columns:
        if col not in future_features.columns:
            future_features[col] = 0  # 缺失的特征值设为0
    
    # 确保列的顺序与训练时一致
    future_features = future_features[features.columns]
    
    # =====================申购金额预测模型=====================
    # 添加常数项(截距)到特征矩阵
    X_purchase = sm.add_constant(features.astype(float))  # 转换为float避免类型错误
    y_purchase = daily_data['total_purchase_amt'].astype(float)  # 目标变量也转换为float
    
    # 使用OLS模型拟合申购金额数据
    # OLS通过最小化残差平方和来估计模型参数
    model_purchase = sm.OLS(y_purchase, X_purchase).fit()
    
    # =====================赎回金额预测模型=====================
    # 添加常数项(截距)到特征矩阵
    X_redeem = sm.add_constant(features.astype(float))
    y_redeem = daily_data['total_redeem_amt'].astype(float)
    
    # 使用OLS模型拟合赎回金额数据
    model_redeem = sm.OLS(y_redeem, X_redeem).fit()
    
    # =====================预测未来30天数据=====================
    # 为预测数据添加常数项
    future_X_purchase = sm.add_constant(future_features.astype(float))
    future_X_redeem = sm.add_constant(future_features.astype(float))
    
    # 根据拟合的模型预测未来30天的申购和赎回金额
    # 预测公式: ŷ = X * β，其中X是特征矩阵，β是模型参数
    future_purchase_pred = model_purchase.predict(future_X_purchase)
    future_redeem_pred = model_redeem.predict(future_X_redeem)
    
    # 构建预测结果DataFrame
    forecast_df = pd.DataFrame({
        'report_date': future_dates,
        'purchase': future_purchase_pred,  # 预测的申购金额
        'redeem': future_redeem_pred       # 预测的赎回金额
    })
    
    # 将日期格式转换为YYYYMMDD字符串格式
    forecast_df['report_date'] = forecast_df['report_date'].dt.strftime('%Y%m%d')
    
    return forecast_df, model_purchase, model_redeem

# 可视化结果函数
def visualize_results(daily_data, forecast_df):
    """
    可视化历史数据和预测结果
    
    参数:
        daily_data: 历史数据
        forecast_df: 预测结果数据
    """
    # 创建一个15x10英寸的图表
    plt.figure(figsize=(15, 10))
    
    # 绘制申购金额图表(上半部分)
    plt.subplot(2, 1, 1)  # 2行1列中的第1个图表
    plt.plot(pd.to_datetime(daily_data['report_date']), daily_data['total_purchase_amt'], label='历史申购金额')
    plt.plot(pd.to_datetime(forecast_df['report_date']), forecast_df['purchase'], label='预测申购金额', linestyle='--')
    plt.title('资金申购金额趋势及预测')
    plt.xlabel('日期')
    plt.ylabel('金额')
    plt.legend()
    plt.grid(True)
    
    # 绘制赎回金额图表(下半部分)
    plt.subplot(2, 1, 2)  # 2行1列中的第2个图表
    plt.plot(pd.to_datetime(daily_data['report_date']), daily_data['total_redeem_amt'], label='历史赎回金额')
    plt.plot(pd.to_datetime(forecast_df['report_date']), forecast_df['redeem'], label='预测赎回金额', linestyle='--')
    plt.title('资金赎回金额趋势及预测')
    plt.xlabel('日期')
    plt.ylabel('金额')
    plt.legend()
    plt.grid(True)
    
    # 调整布局，避免标签重叠
    plt.tight_layout()
    # 保存图表为PNG文件
    plt.savefig('periodic_factor_forecast.png')
    plt.close()  # 关闭图表，释放内存

# 主函数
def main():
    """
    主函数，整合数据读取、处理、建模和预测的完整流程
    """
    # 读取数据
    file_path = './09-时间序列AI大赛(prophet)/04资金流入流出预测/user_balance_table.csv'
    df = load_data(file_path)
    
    # 按日期聚合数据
    daily_data = aggregate_data(df)
    
    # 提取特征(周期因子)
    daily_data, features = extract_features(daily_data)
    
    # 训练模型并预测未来30天
    forecast_df, model_purchase, model_redeem = train_and_predict(daily_data, features)
    
    # 保存预测结果到CSV文件
    forecast_df.to_csv('periodic_factor_forecast_201409.csv', index=False)
    
    # 可视化结果
    visualize_results(daily_data, forecast_df)
    
    # 打印模型摘要信息，显示模型参数和统计显著性
    print("申购金额模型摘要:")
    print(model_purchase.summary())
    print("\n赎回金额模型摘要:")
    print(model_redeem.summary())
    
    print(f"预测结果已保存到 periodic_factor_forecast_201409.csv")
    print(f"可视化结果已保存到 periodic_factor_forecast.png")

# 程序入口
if __name__ == '__main__':
    main()