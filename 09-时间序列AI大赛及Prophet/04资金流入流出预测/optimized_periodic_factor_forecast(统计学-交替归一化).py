"""
    优化的周期因子分析与预测（统计学）
    本脚本通过迭代法（交替归一化/乘法分解）拟合出weekday因子和day因子，
    使得它们的乘积能够最小化历史数据的残差，减少weekday和day之间的干扰，
    并根据优化后的周期因子预测未来30天的资金流入流出趋势。
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
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

# 使用迭代法提取优化的周期因子函数
def extract_optimized_periodic_factors(daily_data, max_iterations=100, tolerance=1e-6):
    """
    使用迭代法（交替归一化/乘法分解）提取优化的周期因子
    
    参数:
        daily_data: 按日期聚合后的数据
        max_iterations: 最大迭代次数
        tolerance: 收敛阈值
    返回:
        daily_data: 添加了周期因子列的数据集
        weekday_factors: 优化后的星期几影响因子
        day_factors: 优化后的日期影响因子
        overall_means: 总平均值
        iteration_history: 迭代历史记录，包含每一步的误差
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
    
    # 初始化因子
    # 首先使用简单的平均值方法作为初始值
    weekday_means = daily_data.groupby('weekday').agg({
        'total_purchase_amt': 'mean',
        'total_redeem_amt': 'mean'
    }).reset_index()
    
    day_means = daily_data.groupby('day_of_month').agg({
        'total_purchase_amt': 'mean',
        'total_redeem_amt': 'mean'
    }).reset_index()
    
    # 初始化为相对于总平均值的比率
    weekday_factors = {
        'purchase': np.array([row['total_purchase_amt'] / overall_means['purchase'] for _, row in weekday_means.iterrows()]),
        'redeem': np.array([row['total_redeem_amt'] / overall_means['redeem'] for _, row in weekday_means.iterrows()])
    }
    
    day_factors = {
        'purchase': {row['day_of_month']: row['total_purchase_amt'] / overall_means['purchase'] for _, row in day_means.iterrows()},
        'redeem': {row['day_of_month']: row['total_redeem_amt'] / overall_means['redeem'] for _, row in day_means.iterrows()}
    }
    
    # 确保day_factors包含所有可能的日期(1-31)
    for day in range(1, 32):
        if day not in day_factors['purchase']:
            day_factors['purchase'][day] = 1.0
        if day not in day_factors['redeem']:
            day_factors['redeem'][day] = 1.0
    
    # 转换为数组以便快速计算
    purchase_day_factors_array = np.array([day_factors['purchase'][d] for d in range(1, 32)])
    redeem_day_factors_array = np.array([day_factors['redeem'][d] for d in range(1, 32)])
    
    # 迭代优化过程
    iteration_history = {}
    
    for target in ['purchase', 'redeem']:
        # 创建对应的目标数组
        target_values = daily_data[f'total_{target}_amt'].values
        # 存储每轮迭代的误差
        errors = []
        
        # 获取对应的因子数组
        if target == 'purchase':
            wd_factors = weekday_factors['purchase'].copy()
            day_factors_array = purchase_day_factors_array.copy()
        else:
            wd_factors = weekday_factors['redeem'].copy()
            day_factors_array = redeem_day_factors_array.copy()
        
        for i in range(max_iterations):
            # 计算当前预测值
            predicted = []
            for _, row in daily_data.iterrows():
                wd = row['weekday']
                day = row['day_of_month']
                predicted_val = overall_means[target] * wd_factors[wd] * day_factors_array[day-1]
                predicted.append(predicted_val)
            predicted = np.array(predicted)
            
            # 计算当前误差
            error = np.mean((target_values - predicted) ** 2)
            errors.append(error)
            
            # 如果误差变化小于阈值，停止迭代
            if i > 0 and abs(errors[i] - errors[i-1]) < tolerance:
                break
            
            # 更新星期几因子
            # 固定日期因子，求解最优的星期几因子
            for wd in range(7):
                # 选择对应星期几的所有数据点
                mask = daily_data['weekday'] == wd
                if np.sum(mask) > 0:
                    # 计算该星期几的平均残差
                    avg_ratio = np.mean(target_values[mask] / (overall_means[target] * day_factors_array[daily_data['day_of_month'][mask]-1]))
                    wd_factors[wd] = avg_ratio
            
            # 归一化星期几因子，使其平均值为1
            wd_factors = wd_factors / np.mean(wd_factors)
            
            # 更新日期因子
            # 固定星期几因子，求解最优的日期因子
            for day in range(1, 32):
                # 选择对应日期的所有数据点
                mask = daily_data['day_of_month'] == day
                if np.sum(mask) > 0:
                    # 计算该日期的平均残差
                    avg_ratio = np.mean(target_values[mask] / (overall_means[target] * wd_factors[daily_data['weekday'][mask]]))
                    day_factors_array[day-1] = avg_ratio
            
            # 归一化日期因子，使其平均值为1
            day_factors_array = day_factors_array / np.mean(day_factors_array)
        
        # 保存优化后的因子
        if target == 'purchase':
            weekday_factors['purchase'] = wd_factors
            purchase_day_factors_array = day_factors_array
        else:
            weekday_factors['redeem'] = wd_factors
            redeem_day_factors_array = day_factors_array
        
        # 保存迭代历史
        iteration_history[target] = errors
    
    # 将数组转回字典格式
    optimized_day_factors = {
        'purchase': {d+1: purchase_day_factors_array[d] for d in range(31)},
        'redeem': {d+1: redeem_day_factors_array[d] for d in range(31)}
    }
    
    # 将weekday_factors也转换为字典格式
    optimized_weekday_factors = {
        'purchase': {wd: weekday_factors['purchase'][wd] for wd in range(7)},
        'redeem': {wd: weekday_factors['redeem'][wd] for wd in range(7)}
    }
    
    return daily_data, optimized_weekday_factors, optimized_day_factors, overall_means, iteration_history

# 预测函数
def predict_future(weekday_factors, day_factors, overall_means):
    """
    使用优化后的周期因子预测未来30天的资金流入流出
    
    参数:
        weekday_factors: 优化后的星期几影响因子
        day_factors: 优化后的日期影响因子
        overall_means: 总平均值
    返回:
        forecast_df: 包含预测结果的DataFrame
        
    预测公式:
    预测值 = 总平均值 × 星期几因子 × 日期因子
    """
    # 准备预测的日期范围（2014-09-01至2014-09-30）
    future_dates = [datetime(2014, 9, i+1) for i in range(30)]
    
    # 为每一天计算预测值
    forecast_data = []
    for date in future_dates:
        weekday = date.weekday()  # 获取星期几(0-6)
        day = date.day           # 获取日期(1-31)
        
        # 获取对应的因子值
        purchase_weekday_factor = weekday_factors['purchase'][weekday]
        redeem_weekday_factor = weekday_factors['redeem'][weekday]
        purchase_day_factor = day_factors['purchase'][day]
        redeem_day_factor = day_factors['redeem'][day]
        
        # 计算预测值: 总平均值 × 星期几因子 × 日期因子
        predicted_purchase = overall_means['purchase'] * purchase_weekday_factor * purchase_day_factor
        predicted_redeem = overall_means['redeem'] * redeem_weekday_factor * redeem_day_factor
        
        forecast_data.append({
            'report_date': date,
            'purchase': predicted_purchase,
            'redeem': predicted_redeem
            # 移除weekday和day_of_month列
        })
    
    # 构建预测结果DataFrame
    forecast_df = pd.DataFrame(forecast_data)
    # 将日期格式转换为YYYYMMDD字符串格式
    forecast_df['report_date'] = forecast_df['report_date'].dt.strftime('%Y%m%d')
    
    return forecast_df

# 可视化结果函数
def visualize_results(daily_data, forecast_df, weekday_factors, day_factors, iteration_history=None):
    """
    可视化历史数据、优化后的周期因子、预测结果和迭代过程
    
    参数:
        daily_data: 历史数据
        forecast_df: 预测结果数据
        weekday_factors: 优化后的星期几影响因子
        day_factors: 优化后的日期影响因子
        iteration_history: 迭代历史记录
    """
    # 创建一个包含多个子图的图表
    fig_size = (20, 20) if iteration_history else (20, 15)
    fig = plt.figure(figsize=fig_size)
    
    # 1. 绘制历史与预测的申购金额图表(折线图)
    ax1 = fig.add_subplot(4, 2, 1) if iteration_history else fig.add_subplot(3, 2, 1)
    ax1.plot(pd.to_datetime(daily_data['report_date']), daily_data['total_purchase_amt'], label='历史申购金额')
    ax1.plot(pd.to_datetime(forecast_df['report_date']), forecast_df['purchase'], label='预测申购金额', linestyle='--')
    ax1.set_title('资金申购金额趋势及预测')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('金额')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 绘制历史与预测的赎回金额图表(折线图)
    ax2 = fig.add_subplot(4, 2, 2) if iteration_history else fig.add_subplot(3, 2, 2)
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
    
    ax3 = fig.add_subplot(4, 2, 3) if iteration_history else fig.add_subplot(3, 2, 3)
    x = np.arange(len(weekdays))
    width = 0.35
    ax3.bar(x - width/2, purchase_factors, width, label='申购因子')
    ax3.bar(x + width/2, redeem_factors, width, label='赎回因子')
    ax3.set_xticks(x)
    ax3.set_xticklabels(weekdays)
    ax3.set_title('优化后的星期几影响因子')
    ax3.set_ylabel('相对于平均值的比率')
    ax3.axhline(y=1.0, color='r', linestyle='-', alpha=0.3)  # 标记平均值线
    ax3.legend()
    
    # 4. 绘制日期的影响因子图表(折线图)
    days = sorted(day_factors['purchase'].keys())
    purchase_day_factors = [day_factors['purchase'][d] for d in days]
    redeem_day_factors = [day_factors['redeem'][d] for d in days]
    
    ax4 = fig.add_subplot(4, 2, 4) if iteration_history else fig.add_subplot(3, 2, 4)
    ax4.plot(days, purchase_day_factors, marker='o', label='申购日期因子')
    ax4.plot(days, redeem_day_factors, marker='s', label='赎回日期因子')
    ax4.set_title('优化后的日期影响因子')
    ax4.set_xlabel('日期(1-31)')
    ax4.set_ylabel('相对于平均值的比率')
    ax4.axhline(y=1.0, color='r', linestyle='-', alpha=0.3)  # 标记平均值线
    ax4.legend()
    ax4.grid(True)
    
    # 5. 绘制预测期间的周期因子分布(散点图)
    ax5 = fig.add_subplot(4, 2, 5) if iteration_history else fig.add_subplot(3, 2, 5)
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
    
    # 6. 如果有迭代历史，绘制误差收敛图
    if iteration_history:
        ax6 = fig.add_subplot(4, 2, 6)
        ax6.plot(iteration_history['purchase'], label='申购误差')
        ax6.plot(iteration_history['redeem'], label='赎回误差')
        ax6.set_title('迭代优化过程中的误差收敛')
        ax6.set_xlabel('迭代次数')
        ax6.set_ylabel('均方误差')
        ax6.set_yscale('log')  # 使用对数刻度以便更好地观察收敛过程
        ax6.legend()
        ax6.grid(True)
        
        # 7. 绘制优化前后的因子对比（如果有原始因子）
        ax7 = fig.add_subplot(4, 2, 7)
        # 这里我们只绘制优化后的因子，因为我们没有原始因子的数据
        # 在实际应用中，可以对比优化前后的因子变化
        ax7.axis('off')
        ax7.text(0.5, 0.5, '因子优化完成\n已移除weekday和day之间的相互干扰', 
                 horizontalalignment='center', verticalalignment='center', 
                 fontsize=12)
    
    # 调整布局，避免标签重叠
    plt.tight_layout()
    # 保存图表为PNG文件
    plt.savefig('optimized_periodic_factor_forecast.png')
    plt.close()  # 关闭图表，释放内存

# 计算拟合优度函数
def calculate_accuracy(daily_data, weekday_factors, day_factors, overall_means):
    """
    计算模型的拟合优度
    
    参数:
        daily_data: 历史数据
        weekday_factors: 优化后的星期几影响因子
        day_factors: 优化后的日期影响因子
        overall_means: 总平均值
    返回:
        accuracy_metrics: 包含各种评估指标的字典
    """
    # 计算历史数据的预测值
    predicted_purchase = []
    predicted_redeem = []
    
    for _, row in daily_data.iterrows():
        wd = row['weekday']
        day = row['day_of_month']
        
        # 计算预测值
        pred_purchase = overall_means['purchase'] * weekday_factors['purchase'][wd] * day_factors['purchase'][day]
        pred_redeem = overall_means['redeem'] * weekday_factors['redeem'][wd] * day_factors['redeem'][day]
        
        predicted_purchase.append(pred_purchase)
        predicted_redeem.append(pred_redeem)
    
    # 转换为numpy数组
    predicted_purchase = np.array(predicted_purchase)
    predicted_redeem = np.array(predicted_redeem)
    actual_purchase = daily_data['total_purchase_amt'].values
    actual_redeem = daily_data['total_redeem_amt'].values
    
    # 计算评估指标
    def calculate_metrics(actual, predicted):
        # 计算均方误差
        mse = np.mean((actual - predicted) ** 2)
        # 计算均方根误差
        rmse = np.sqrt(mse)
        # 计算平均绝对误差
        mae = np.mean(np.abs(actual - predicted))
        # 计算R²值（决定系数）
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        ss_res = np.sum((actual - predicted) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
    
    accuracy_metrics = {
        'purchase': calculate_metrics(actual_purchase, predicted_purchase),
        'redeem': calculate_metrics(actual_redeem, predicted_redeem)
    }
    
    return accuracy_metrics

# 主函数
def main():
    """
    主函数，整合数据读取、处理、周期因子优化和预测的完整流程
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
    
    # 提取优化后的周期因子
    daily_data, weekday_factors, day_factors, overall_means, iteration_history = extract_optimized_periodic_factors(daily_data)
    
    # 预测未来30天数据
    forecast_df = predict_future(weekday_factors, day_factors, overall_means)
    
    # 计算模型的拟合优度
    accuracy_metrics = calculate_accuracy(daily_data, weekday_factors, day_factors, overall_means)
    
    # 保存预测结果到CSV文件
    forecast_df.to_csv('optimized_periodic_factor_forecast_201409.csv', index=False)
    
    # 可视化结果
    visualize_results(daily_data, forecast_df, weekday_factors, day_factors, iteration_history)
    
    # 打印周期因子信息
    print("优化后的星期几影响因子:")
    weekdays = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    for i, day in enumerate(weekdays):
        print(f"  {day}: 申购因子={weekday_factors['purchase'][i]:.4f}, 赎回因子={weekday_factors['redeem'][i]:.4f}")
    
    print("\n优化后的日期影响因子(前10天):")
    days = sorted(day_factors['purchase'].keys())[:10]
    for day in days:
        print(f"  {day}号: 申购因子={day_factors['purchase'][day]:.4f}, 赎回因子={day_factors['redeem'][day]:.4f}")
    
    print("\n总平均值:")
    print(f"  平均申购金额: {overall_means['purchase']:.2f}")
    print(f"  平均赎回金额: {overall_means['redeem']:.2f}")
    
    # 打印模型评估指标
    print("\n模型评估指标:")
    for target in ['purchase', 'redeem']:
        target_name = '申购' if target == 'purchase' else '赎回'
        print(f"  {target_name}:")
        print(f"    均方误差(MSE): {accuracy_metrics[target]['mse']:.2f}")
        print(f"    均方根误差(RMSE): {accuracy_metrics[target]['rmse']:.2f}")
        print(f"    平均绝对误差(MAE): {accuracy_metrics[target]['mae']:.2f}")
        print(f"    决定系数(R²): {accuracy_metrics[target]['r2']:.4f}")
    
    print(f"\n预测结果已保存到 optimized_periodic_factor_forecast_201409.csv")
    print(f"可视化结果已保存到 optimized_periodic_factor_forecast.png")

# 程序入口
if __name__ == '__main__':
    main()