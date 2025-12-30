import pandas as pd
import matplotlib.pyplot as plt

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 读取数据
df = pd.read_csv('./09-时间序列AI大赛(prophet)/04资金流入流出预测/user_balance_table.csv', encoding='utf-8')

# 转换 report_date 为日期格式
df['report_date'] = pd.to_datetime(df['report_date'], format='%Y%m%d')

# 截取 2014-03 到 2014-08 的数据
mask = (df['report_date'] >= '2014-03-01') & (df['report_date'] <= '2014-08-31')
df_period = df.loc[mask]

# 按 report_date 聚合申购和赎回金额
daily_data = df_period.groupby('report_date')[['total_purchase_amt', 'total_redeem_amt']].sum().reset_index()

# 添加星期几列（0=周一，6=周日）
daily_data['weekday'] = daily_data['report_date'].dt.weekday

# 创建星期几的中文标签映射
weekday_labels = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五', 5: '周六', 6: '周日'}

# 按星期几分组计算均值
weekday_avg = daily_data.groupby('weekday').agg({
    'total_purchase_amt': 'mean',
    'total_redeem_amt': 'mean'
}).reset_index()

# 重命名列名更清晰
weekday_avg.rename(columns={
    'total_purchase_amt': '平均申购金额',
    'total_redeem_amt': '平均赎回金额'
}, inplace=True)

# 映射星期几的中文标签
weekday_avg['星期'] = weekday_avg['weekday'].map(weekday_labels)

# 按周一到周日的顺序排序
weekday_order = [0, 1, 2, 3, 4, 5, 6]
weekday_avg = weekday_avg.set_index('weekday').loc[weekday_order].reset_index()

# 打印结果
print("按星期几统计的平均资金流入流出情况：")
print(weekday_avg[['星期', '平均申购金额', '平均赎回金额']])

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(weekday_avg['星期'], weekday_avg['平均申购金额'], marker='o', label='平均申购金额', linewidth=2)
plt.plot(weekday_avg['星期'], weekday_avg['平均赎回金额'], marker='s', label='平均赎回金额', linewidth=2)

plt.xlabel('星期', fontsize=12)
plt.ylabel('金额', fontsize=12)
plt.title('2014-03到2014-08按星期几统计的平均资金流入流出情况', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# 保存图表
plt.savefig('weekday_analysis.png', dpi=300)
plt.show()

# 输出结果到CSV文件
output_columns = ['weekday', '星期', '平均申购金额', '平均赎回金额']
weekday_avg[output_columns].to_csv('weekday_avg_data.csv', index=False, encoding='utf-8-sig')
print('按星期几统计的平均资金流入流出数据已保存到 weekday_avg_data.csv')
print('图表已保存到 weekday_analysis.png')

# ===== 新增：按1-31日统计平均资金流入流出 =====

# 添加日期中的日（day of month）
daily_data['day_of_month'] = daily_data['report_date'].dt.day

# 按日分组计算均值
day_avg = daily_data.groupby('day_of_month').agg({
    'total_purchase_amt': 'mean',
    'total_redeem_amt': 'mean'
}).reset_index()

# 重命名列名更清晰
day_avg.rename(columns={
    'total_purchase_amt': '平均申购金额',
    'total_redeem_amt': '平均赎回金额'
}, inplace=True)

# 按1-31日顺序排序（确保所有日期都包含）
all_days = pd.DataFrame({'day_of_month': range(1, 32)})
day_avg = pd.merge(all_days, day_avg, on='day_of_month', how='left').fillna(0)

# 打印结果
print("\n按1-31日统计的平均资金流入流出情况：")
print(day_avg[['day_of_month', '平均申购金额', '平均赎回金额']])

# 绘制折线图
plt.figure(figsize=(12, 6))
plt.plot(day_avg['day_of_month'], day_avg['平均申购金额'], marker='o', label='平均申购金额', linewidth=2)
plt.plot(day_avg['day_of_month'], day_avg['平均赎回金额'], marker='s', label='平均赎回金额', linewidth=2)

plt.xlabel('日期（1-31日）', fontsize=12)
plt.ylabel('金额', fontsize=12)
plt.title('2014-03到2014-08按1-31日统计的平均资金流入流出情况', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(1, 32))  # 显示1-31所有日期的刻度
plt.tight_layout()

# 保存图表
plt.savefig('day_of_month_analysis.png', dpi=300)
plt.show()

# 输出结果到CSV文件
day_output_columns = ['day_of_month', '平均申购金额', '平均赎回金额']
day_avg[day_output_columns].to_csv('day_of_month_avg_data.csv', index=False, encoding='utf-8-sig')
print('按1-31日统计的平均资金流入流出数据已保存到 day_of_month_avg_data.csv')
print('图表已保存到 day_of_month_analysis.png')