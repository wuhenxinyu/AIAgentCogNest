"""
利率预测模型(机器学习OLS最小二乘法线性回归)
本脚本通过统计分析和可视化展示利率数据，构建包含以下重点的预测模型：
1. mfd_7daily_yield与1W SHIBOR的滞后效应
2. 节假日前后的收益率调整因子
3. 分析利率变动对用户申购赎回行为的影响

本脚本使用最小二乘法线性回归模型进行预测，输入特征包括：
- 利率历史数据及滞后项
- 节假日效应因子
- 季节性因子
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime, timedelta
import os
from sklearn.metrics import r2_score, mean_squared_error

# 设置中文字体，确保图表中的中文能正常显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 定义2014年节假日，用于创建节假日因子
holidays_2014 = {
    '元旦': ['2014-01-01'],
    '春节': ['2014-01-31', '2014-02-01', '2014-02-02', '2014-02-03', '2014-02-04', '2014-02-05', '2014-02-06'],
    '清明节': ['2014-04-05', '2014-04-06', '2014-04-07'],
    '劳动节': ['2014-05-01', '2014-05-02', '2014-05-03'],
    '端午节': ['2014-06-02'],
    '中秋节': ['2014-09-08'],
    '国庆节': ['2014-10-01', '2014-10-02', '2014-10-03', '2014-10-04', '2014-10-05', '2014-10-06', '2014-10-07']
}

# 读取数据函数
def load_interest_data(mfd_file, shibor_file, user_file):
    """
    读取利率相关的三个CSV文件，并进行预处理
    
    参数:
        mfd_file: 货币基金收益率文件路径
        shibor_file: SHIBOR利率文件路径
        user_file: 用户余额表文件路径
    返回:
        merged_df: 合并后的DataFrame，包含所有相关数据
    """
    # 读取货币基金收益率数据
    mfd_df = pd.read_csv(mfd_file)
    mfd_df['mfd_date'] = pd.to_datetime(mfd_df['mfd_date'], format='%Y%m%d')
    
    # 读取SHIBOR利率数据
    shibor_df = pd.read_csv(shibor_file)
    shibor_df['mfd_date'] = pd.to_datetime(shibor_df['mfd_date'], format='%Y%m%d')
    
    # 读取用户交易数据（只读取需要的列和部分数据以提高效率）
    user_df = pd.read_csv(user_file, usecols=['report_date', 'total_purchase_amt', 'total_redeem_amt'])
    user_df['report_date'] = pd.to_datetime(user_df['report_date'], format='%Y%m%d')
    
    # 按日期聚合用户交易数据
    user_daily = user_df.groupby('report_date').agg({
        'total_purchase_amt': 'sum',
        'total_redeem_amt': 'sum'
    }).reset_index()
    
    # 合并货币基金数据和SHIBOR数据
    merged_df = pd.merge(mfd_df, shibor_df, on='mfd_date', how='inner')
    
    # 重命名用户数据的日期列，以便合并
    user_daily = user_daily.rename(columns={'report_date': 'mfd_date'})
    
    # 合并用户数据
    merged_df = pd.merge(merged_df, user_daily, on='mfd_date', how='left')
    
    # 按日期排序
    merged_df = merged_df.sort_values('mfd_date')
    
    return merged_df

# 提取特征函数
def extract_features(df):
    """
    提取利率预测模型所需的特征，包括：
    - mfd_7daily_yield和1W SHIBOR的滞后效应
    - 节假日前后的收益率调整因子
    - 周期性因子（星期几、月份）
    
    参数:
        df: 包含原始数据的DataFrame
    返回:
        df: 添加了特征列的数据集
        features: 用于建模的特征矩阵
    """
    # 复制数据，避免修改原始数据
    df_copy = df.copy()
    
    # 提取星期几和月份特征
    df_copy['weekday'] = df_copy['mfd_date'].dt.weekday
    df_copy['month'] = df_copy['mfd_date'].dt.month
    
    # 创建滞后效应特征（滞后1-3天）
    for lag in range(1, 4):
        df_copy[f'mfd_7daily_yield_lag{lag}'] = df_copy['mfd_7daily_yield'].shift(lag)
        df_copy[f'shibor_1w_lag{lag}'] = df_copy['Interest_1_W'].shift(lag)  # 修复列名
    
    # 创建节假日因子
    df_copy['is_holiday'] = 0
    df_copy['holiday_eve'] = 0
    df_copy['holiday_after'] = 0
    
    all_holidays = []
    for holiday_dates in holidays_2014.values():
        all_holidays.extend(holiday_dates)
    
    all_holidays = pd.to_datetime(all_holidays)
    
    # 标记节假日、节前和节后
    for idx, row in df_copy.iterrows():
        current_date = row['mfd_date']
        
        # 检查是否为节假日
        if current_date in all_holidays:
            df_copy.loc[idx, 'is_holiday'] = 1
        
        # 检查是否为节假日的前一天
        holiday_eve_date = current_date + timedelta(days=1)
        if holiday_eve_date in all_holidays:
            df_copy.loc[idx, 'holiday_eve'] = 1
        
        # 检查是否为节假日的后一天
        holiday_after_date = current_date - timedelta(days=1)
        if holiday_after_date in all_holidays:
            df_copy.loc[idx, 'holiday_after'] = 1
    
    # 创建星期几的哑变量
    weekday_dummies = pd.get_dummies(df_copy['weekday'], prefix='weekday', drop_first=True)
    
    # 创建月份的哑变量
    month_dummies = pd.get_dummies(df_copy['month'], prefix='month', drop_first=True)
    
    # 选择用于建模的特征
    feature_cols = [col for col in df_copy.columns if col.startswith('mfd_7daily_yield_lag') or 
                    col.startswith('shibor_1w_lag')] + ['is_holiday', 'holiday_eve', 'holiday_after']
    
    # 合并所有特征
    features = pd.concat([df_copy[feature_cols], weekday_dummies, month_dummies], axis=1)
    
    # 删除包含NaN的行（由于滞后效应会产生NaN）
    df_copy = df_copy.dropna()
    features = features.dropna()
    
    return df_copy, features

# 训练模型并预测函数
def train_and_predict(df, features):
    """
    使用OLS线性回归模型，基于提取的特征预测未来7天的利率
    
    参数:
        df: 包含历史数据的DataFrame
        features: 特征矩阵
    返回:
        forecast_df: 包含预测结果的DataFrame
        model: 训练好的利率预测模型
    """
    # 准备目标变量（mfd_7daily_yield）
    y = df['mfd_7daily_yield']
    
    # 确保特征和目标变量的行数一致
    min_length = min(len(features), len(y))
    features = features.iloc[:min_length]
    y = y.iloc[:min_length]
    
    # 手动添加截距项，避免使用sm.add_constant可能导致的维度问题
    X_with_intercept = features.copy()
    X_with_intercept['intercept'] = 1
    
    # 使用OLS模型拟合数据
    model = sm.OLS(y, X_with_intercept.astype(float)).fit()
    
    # 准备预测的日期范围（未来7天）
    last_date = df['mfd_date'].max()
    future_dates = [last_date + timedelta(days=i+1) for i in range(7)]
    
    # 为预测数据创建特征，传入训练时的特征列名以确保一致性
    future_features = create_future_features(df, future_dates, features.columns)
    
    # 为预测数据手动添加截距项，保持与训练时一致
    future_features_with_intercept = future_features.copy()
    future_features_with_intercept['intercept'] = 1
    
    # 进行预测，确保特征列顺序与训练时一致
    future_predictions = model.predict(future_features_with_intercept[X_with_intercept.columns].astype(float))
    
    # 构建预测结果DataFrame
    forecast_df = pd.DataFrame({
        'mfd_date': future_dates,
        'predicted_mfd_7daily_yield': future_predictions
    })
    
    # 将日期格式转换为YYYYMMDD字符串格式
    forecast_df['mfd_date'] = forecast_df['mfd_date'].dt.strftime('%Y%m%d')
    
    return forecast_df, model

# 创建未来数据的特征
def create_future_features(df, future_dates, feature_columns):
    """
    为未来日期创建特征数据
    
    参数:
        df: 历史数据
        future_dates: 未来要预测的日期列表
        feature_columns: 训练时使用的特征列名，确保预测时特征一致
    返回:
        future_features: 未来日期的特征矩阵，与训练时特征维度一致
    """
    future_df = pd.DataFrame({'mfd_date': future_dates})
    
    # 提取基本时间特征
    future_df['weekday'] = future_df['mfd_date'].dt.weekday
    future_df['month'] = future_df['mfd_date'].dt.month
    
    # 获取最新的历史数据用于创建滞后特征
    latest_data = df.iloc[-3:].copy()
    
    # 创建滞后效应特征
    for i, date in enumerate(future_dates):
        for lag in range(1, 4):
            # 找到对应滞后日期的数据，确保索引在有效范围内
            lag_idx = -lag
            # 如果没有足够的历史数据，使用平均值
            if lag_idx < -len(latest_data):
                future_df.loc[i, f'mfd_7daily_yield_lag{lag}'] = latest_data['mfd_7daily_yield'].mean()
                future_df.loc[i, f'shibor_1w_lag{lag}'] = latest_data['Interest_1_W'].mean()
            else:
                future_df.loc[i, f'mfd_7daily_yield_lag{lag}'] = latest_data.iloc[lag_idx]['mfd_7daily_yield']
                future_df.loc[i, f'shibor_1w_lag{lag}'] = latest_data.iloc[lag_idx]['Interest_1_W']
    
    # 创建节假日因子
    future_df['is_holiday'] = 0
    future_df['holiday_eve'] = 0
    future_df['holiday_after'] = 0
    
    all_holidays = []
    for holiday_dates in holidays_2014.values():
        all_holidays.extend(holiday_dates)
    
    all_holidays = pd.to_datetime(all_holidays)
    
    # 标记节假日、节前和节后
    for idx, row in future_df.iterrows():
        current_date = row['mfd_date']
        
        # 检查是否为节假日
        if current_date in all_holidays:
            future_df.loc[idx, 'is_holiday'] = 1
        
        # 检查是否为节假日的前一天
        holiday_eve_date = current_date + timedelta(days=1)
        if holiday_eve_date in all_holidays:
            future_df.loc[idx, 'holiday_eve'] = 1
        
        # 检查是否为节假日的后一天
        holiday_after_date = current_date - timedelta(days=1)
        if holiday_after_date in all_holidays:
            future_df.loc[idx, 'holiday_after'] = 1
    
    # 创建星期几的哑变量，保持与训练时一致（drop_first=True）
    weekday_dummies = pd.get_dummies(future_df['weekday'], prefix='weekday', drop_first=True)
    
    # 创建月份的哑变量，保持与训练时一致（drop_first=True）
    month_dummies = pd.get_dummies(future_df['month'], prefix='month', drop_first=True)
    
    # 合并所有特征
    all_features = pd.concat([future_df, weekday_dummies, month_dummies], axis=1)
    
    # 确保与训练时的特征列完全一致
    # 创建一个空的特征矩阵，列名与训练时相同
    future_features = pd.DataFrame(index=future_df.index)
    
    for col in feature_columns:
        if col in all_features.columns:
            future_features[col] = all_features[col]
        else:
            # 如果未来特征中没有该列，填充0
            future_features[col] = 0
    
    return future_features

# 分析利率变动对用户行为的影响
def analyze_interest_impact(df):
    """
    分析利率变动对用户申购赎回行为的影响
    
    参数:
        df: 包含利率和用户行为数据的DataFrame
    返回:
        correlation_results: 相关性分析结果
    """
    # 计算利率变动
    df['interest_change'] = df['mfd_7daily_yield'].diff()
    
    # 计算用户行为指标
    df['net_flow'] = df['total_purchase_amt'] - df['total_redeem_amt']
    df['purchase_redeem_ratio'] = df['total_purchase_amt'] / (df['total_redeem_amt'] + 1)  # +1避免除零错误
    
    # 计算相关性
    correlation_results = {
        'interest_purchase_corr': df['mfd_7daily_yield'].corr(df['total_purchase_amt']),
        'interest_redeem_corr': df['mfd_7daily_yield'].corr(df['total_redeem_amt']),
        'interest_net_flow_corr': df['mfd_7daily_yield'].corr(df['net_flow']),
        'change_purchase_corr': df['interest_change'].corr(df['total_purchase_amt']),
        'change_redeem_corr': df['interest_change'].corr(df['total_redeem_amt']),
        'change_net_flow_corr': df['interest_change'].corr(df['net_flow'])
    }
    
    return correlation_results

# 可视化结果函数
def visualize_results(df, forecast_df, correlation_results):
    """
    可视化利率历史数据、预测结果以及利率与用户行为的关系
    
    参数:
        df: 历史数据
        forecast_df: 预测结果数据
        correlation_results: 相关性分析结果
    """
    # 创建一个20x15英寸的图表
    plt.figure(figsize=(20, 15))
    
    # 绘制利率历史和预测图表
    plt.subplot(3, 1, 1)
    plt.plot(df['mfd_date'], df['mfd_7daily_yield'], label='历史7日年化收益率')
    plt.plot(pd.to_datetime(forecast_df['mfd_date']), forecast_df['predicted_mfd_7daily_yield'], 
             label='预测7日年化收益率', linestyle='--')
    plt.title('货币基金7日年化收益率趋势及预测')
    plt.xlabel('日期')
    plt.ylabel('收益率(%)')
    plt.legend()
    plt.grid(True)
    
    # 绘制SHIBOR利率图表
    plt.subplot(3, 1, 2)
    plt.plot(df['mfd_date'], df['Interest_1_W'], label='1W SHIBOR')  # 修改为正确的列名'Interest_1_W'
    plt.title('1W SHIBOR利率趋势')
    plt.xlabel('日期')
    plt.ylabel('利率(%)')
    plt.legend()
    plt.grid(True)
    
    # 绘制利率与用户行为关系图表
    plt.subplot(3, 1, 3)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    ax1.plot(df['mfd_date'], df['mfd_7daily_yield'], 'b-', label='7日年化收益率')
    ax2.plot(df['mfd_date'], df['net_flow'], 'r-', label='资金净流入')
    
    plt.title('利率与资金净流入关系（相关系数: {:.3f}）'.format(correlation_results['interest_net_flow_corr']))
    ax1.set_xlabel('日期')
    ax1.set_ylabel('收益率(%)', color='b')
    ax2.set_ylabel('资金净流入', color='r')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.grid(True)
    
    # 调整布局，避免标签重叠
    plt.tight_layout()
    # 保存图表为PNG文件
    plt.savefig('interest_rate_forecast.png')
    plt.close()  # 关闭图表，释放内存

# 主函数
def main():
    """
    主函数，整合数据读取、处理、建模、预测和分析的完整流程
    """
    # 定义文件路径
    mfd_file = '/Users/clz/Downloads/xmkf/llm/AIAgentCogNest/09-时间序列AI大赛(Prophet)/04资金流入流出预测/mfd_day_share_interest.csv'
    shibor_file = '/Users/clz/Downloads/xmkf/llm/AIAgentCogNest/09-时间序列AI大赛(Prophet)/04资金流入流出预测/mfd_bank_shibor.csv'
    user_file = '/Users/clz/Downloads/xmkf/llm/AIAgentCogNest/09-时间序列AI大赛(Prophet)/04资金流入流出预测/user_balance_table.csv'
    
    # 读取数据
    print("正在读取数据...")
    df = load_interest_data(mfd_file, shibor_file, user_file)
    
    # 提取特征
    print("正在提取特征...")
    df_with_features, features = extract_features(df)
    
    # 训练模型并预测未来7天
    print("正在训练模型并进行预测...")
    forecast_df, model = train_and_predict(df_with_features, features)
    
    # 分析利率变动对用户行为的影响
    print("正在分析利率对用户行为的影响...")
    correlation_results = analyze_interest_impact(df_with_features)
    
    # 保存预测结果到CSV文件
    forecast_df.to_csv('interest_rate_forecast_result.csv', index=False)
    
    # 可视化结果
    print("正在生成可视化结果...")
    visualize_results(df_with_features, forecast_df, correlation_results)
    
    # 打印模型摘要信息
    print("\n利率预测模型摘要:")
    print(model.summary())
    
    # 打印相关性分析结果
    print("\n利率与用户行为相关性分析:")
    for key, value in correlation_results.items():
        print(f"{key}: {value:.4f}")
    
    print("\n预测结果已保存到 interest_rate_forecast_result.csv")
    print("可视化结果已保存到 interest_rate_forecast.png")

# 程序入口
if __name__ == '__main__':
    main()