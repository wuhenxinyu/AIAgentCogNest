"""
    简单的周期因子分析与预测（统计学）
    本脚本通过统计分析和可视化展示2014年3月至2014年8月的用户资金流入流出数据，
    并根据星期几和日期的周期因子，预测未来30天的资金流入流出趋势。

    用周期因子 weekday 和 day进行计算。即找到 weekday和day的规律，然后针对未来的30天，看属于weekday和day的情况进行乘法即可
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# 设置中文字体，确保图表中的中文能正常显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']  # 使用黑体显示中文
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

# 提取周期因子函数
def extract_periodic_factors(daily_data):
    """
    提取周期因子：星期几和日期的影响因子
    
    参数:
        daily_data: 按日期聚合后的数据
    返回:
        daily_data: 添加了周期因子列的数据集
        weekday_factors: 星期几的影响因子
        day_factors: 日期的影响因子
        overall_means: 总平均值
        
    周期因子计算原理：
    1. 计算每个周期因子(星期几/日期)的平均值与总平均值的比率
    2. 例如：如果周一的平均申购金额是总平均的1.2倍，则周一的申购因子为1.2
    3. 未来预测时，将对应周期因子相乘再乘以总平均值得到预测值
    """
    # 提取星期几特征(0=周一, 1=周二, ..., 6=周日)
    daily_data['weekday'] = daily_data['report_date'].dt.weekday
    # 提取日期中的几号特征(1-31)
    daily_data['day_of_month'] = daily_data['report_date'].dt.day
    
    # 计算总平均值
    overall_means = {
        'purchase': daily_data['total_purchase_amt'].mean(),
        'redeem': daily_data['total_redeem_amt'].mean()
    }
    
    # 计算星期几的影响因子
    # 1. 按星期几分组计算平均值
    # 2. 除以总平均值得到影响因子
    weekday_means = daily_data.groupby('weekday').agg({
        'total_purchase_amt': 'mean',
        'total_redeem_amt': 'mean'
    }).reset_index()
    
    # 计算星期几的影响因子（相对于总平均值的比率）
    weekday_factors = {
        'purchase': {row['weekday']: row['total_purchase_amt'] / overall_means['purchase'] 
                     for _, row in weekday_means.iterrows()},
        'redeem': {row['weekday']: row['total_redeem_amt'] / overall_means['redeem'] 
                   for _, row in weekday_means.iterrows()}
    }
    
    # 计算日期的影响因子
    # 1. 按日期分组计算平均值
    # 2. 除以总平均值得到影响因子
    day_means = daily_data.groupby('day_of_month').agg({
        'total_purchase_amt': 'mean',
        'total_redeem_amt': 'mean'
    }).reset_index()
    
    # 计算日期的影响因子（相对于总平均值的比率）
    day_factors = {
        'purchase': {row['day_of_month']: row['total_purchase_amt'] / overall_means['purchase'] 
                     for _, row in day_means.iterrows()},
        'redeem': {row['day_of_month']: row['total_redeem_amt'] / overall_means['redeem'] 
                   for _, row in day_means.iterrows()}
    }
    
    return daily_data, weekday_factors, day_factors, overall_means

# 预测函数
def predict_future(weekday_factors, day_factors, overall_means):
    """
    使用周期因子预测未来30天的资金流入流出
    
    参数:
        weekday_factors: 星期几的影响因子
        day_factors: 日期的影响因子
        overall_means: 总平均值
    返回:
        forecast_df: 包含预测结果的DataFrame
        
    预测公式:
    预测值 = 总平均值 × 星期几因子 × 日期因子
    例如: 2014-09-01的申购预测 = 总平均申购 × 周一因子 × 1号因子
    """
    # 准备预测的日期范围（2014-09-01至2014-09-30）
    future_dates = [datetime(2014, 9, i+1) for i in range(30)]
    
    # 为每一天计算预测值
    forecast_data = []
    for date in future_dates:
        weekday = date.weekday()  # 获取星期几(0-6)
        day = date.day           # 获取日期(1-31)
        
        # 如果某天的日期因子不存在（例如31号在训练数据中没有出现过），则使用默认值1.0
        purchase_day_factor = day_factors['purchase'].get(day, 1.0)
        redeem_day_factor = day_factors['redeem'].get(day, 1.0)
        
        # 计算预测值: 总平均值 × 星期几因子 × 日期因子
        predicted_purchase = overall_means['purchase'] * weekday_factors['purchase'][weekday] * purchase_day_factor
        predicted_redeem = overall_means['redeem'] * weekday_factors['redeem'][weekday] * redeem_day_factor
        
        forecast_data.append({
            'report_date': date,
            'purchase': predicted_purchase,
            'redeem': predicted_redeem,
            'weekday': weekday,
            'day_of_month': day
        })
    
    # 构建预测结果DataFrame
    forecast_df = pd.DataFrame(forecast_data)
    # 将日期格式转换为YYYYMMDD字符串格式
    forecast_df['report_date'] = forecast_df['report_date'].dt.strftime('%Y%m%d')
    
    return forecast_df

# 可视化结果函数
def visualize_results(daily_data, forecast_df, weekday_factors, day_factors):
    """
    可视化历史数据、周期因子和预测结果
    
    参数:
        daily_data: 历史数据
        forecast_df: 预测结果数据
        weekday_factors: 星期几的影响因子
        day_factors: 日期的影响因子
    """
    # 创建一个包含多个子图的图表
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 绘制历史与预测的申购金额图表(折线图)
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(pd.to_datetime(daily_data['report_date']), daily_data['total_purchase_amt'], label='历史申购金额')
    ax1.plot(pd.to_datetime(forecast_df['report_date']), forecast_df['purchase'], label='预测申购金额', linestyle='--')
    ax1.set_title('资金申购金额趋势及预测')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('金额')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 绘制历史与预测的赎回金额图表(折线图)
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(pd.to_datetime(daily_data['report_date']), daily_data['total_redeem_amt'], label='历史赎回金额')
    ax2.plot(pd.to_datetime(forecast_df['report_date']), forecast_df['redeem'], label='预测赎回金额', linestyle='--')
    ax2.set_title('资金赎回金额趋势及预测')
    ax2.set_xlabel('日期')
    ax2.set_ylabel('金额')
    ax2.legend()
    ax2.grid(True)
    
    # 3. 绘制星期几的影响因子图表(柱状图)
    weekdays = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    purchase_factors = [weekday_factors['purchase'][i] for i in range(7)]
    redeem_factors = [weekday_factors['redeem'][i] for i in range(7)]
    
    ax3 = fig.add_subplot(3, 2, 3)
    x = np.arange(len(weekdays))
    width = 0.35
    ax3.bar(x - width/2, purchase_factors, width, label='申购因子')
    ax3.bar(x + width/2, redeem_factors, width, label='赎回因子')
    ax3.set_xticks(x)
    ax3.set_xticklabels(weekdays)
    ax3.set_title('星期几影响因子')
    ax3.set_ylabel('相对于平均值的比率')
    ax3.axhline(y=1.0, color='r', linestyle='-', alpha=0.3)  # 标记平均值线
    ax3.legend()
    
    # 4. 绘制日期的影响因子图表(折线图)
    days = sorted(day_factors['purchase'].keys())
    purchase_day_factors = [day_factors['purchase'][d] for d in days]
    redeem_day_factors = [day_factors['redeem'][d] for d in days]
    
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(days, purchase_day_factors, marker='o', label='申购日期因子')
    ax4.plot(days, redeem_day_factors, marker='s', label='赎回日期因子')
    ax4.set_title('日期影响因子')
    ax4.set_xlabel('日期(1-31)')
    ax4.set_ylabel('相对于平均值的比率')
    ax4.axhline(y=1.0, color='r', linestyle='-', alpha=0.3)  # 标记平均值线
    ax4.legend()
    ax4.grid(True)
    
    # 5. 绘制预测期间的周期因子分布(散点图)
    ax5 = fig.add_subplot(3, 2, 5)
    forecast_dates = pd.to_datetime(forecast_df['report_date'])
    forecast_weekdays = forecast_dates.dt.weekday
    forecast_weekday_labels = [weekdays[wd] for wd in forecast_weekdays]
    
    # 使用不同颜色表示不同星期几
    colors = plt.cm.tab10(np.linspace(0, 1, 7))
    color_map = {i: colors[i] for i in range(7)}
    
    ax5.scatter(forecast_dates, forecast_df['purchase'], 
               c=[color_map[wd] for wd in forecast_weekdays], 
               label='预测申购金额', alpha=0.7)
    ax5.set_title('9月预测申购金额 - 按星期几分类')
    ax5.set_xlabel('日期')
    ax5.set_ylabel('金额')
    # 创建自定义图例
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[i], label=weekdays[i]) for i in range(7)]
    ax5.legend(handles=handles, title='星期几')
    ax5.grid(True)
    
    # 调整布局，避免标签重叠
    plt.tight_layout()
    # 保存图表为PNG文件
    plt.savefig('simple_periodic_factor_forecast.png')
    plt.close()  # 关闭图表，释放内存

# 主函数
def main():
    """
    主函数，整合数据读取、处理、周期因子计算和预测的完整流程
    """
    # 读取数据
    file_path = './09-时间序列AI大赛(prophet)/04资金流入流出预测/user_balance_table.csv'  # 使用相对路径
    df = load_data(file_path)
    
    # 按日期聚合数据
    daily_data = aggregate_data(df)
    
    # 提取周期因子
    daily_data, weekday_factors, day_factors, overall_means = extract_periodic_factors(daily_data)
    
    # 预测未来30天数据
    forecast_df = predict_future(weekday_factors, day_factors, overall_means)
    
    # 保存预测结果到CSV文件
    forecast_df.to_csv('simple_periodic_factor_forecast_201409.csv', index=False)
    
    # 可视化结果
    visualize_results(daily_data, forecast_df, weekday_factors, day_factors)
    
    # 打印周期因子信息
    print("星期几影响因子:")
    weekdays = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    for i, day in enumerate(weekdays):
        print(f"  {day}: 申购因子={weekday_factors['purchase'][i]:.4f}, 赎回因子={weekday_factors['redeem'][i]:.4f}")
    
    print("\n总平均值:")
    print(f"  平均申购金额: {overall_means['purchase']:.2f}")
    print(f"  平均赎回金额: {overall_means['redeem']:.2f}")
    
    print(f"\n预测结果已保存到 simple_periodic_factor_forecast_201409.csv")
    print(f"可视化结果已保存到 simple_periodic_factor_forecast.png")

# 程序入口
if __name__ == '__main__':
    main()