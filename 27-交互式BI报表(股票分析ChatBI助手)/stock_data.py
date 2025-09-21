import tushare as ts
import pandas as pd
from datetime import datetime
import os

# 设置tushare token
# 注意：使用前需要先注册tushare账号并获取token
ts.set_token('91ff05c57927e99826a17d718b1f95180c82b36cf2435d50bfbbc942')  # 请替换为你的token
pro = ts.pro_api()

# 定义股票代码列表
stock_codes = {
    '贵州茅台': '600519.SH',
    '五粮液': '000858.SZ',
    '国泰君安': '601211.SH',
    '中芯国际': '688981.SH'
}

# 获取当前日期
end_date = datetime.now().strftime('%Y%m%d')

# 创建DataFrame列表用于存储所有股票数据
all_stocks_data = []

# 获取每只股票的数据
for stock_name, stock_code in stock_codes.items():
    try:
        # 获取日线数据
        df = pro.daily(ts_code=stock_code, 
                      start_date='20200101',
                      end_date=end_date)
        
        # 添加股票名称列
        df['stock_name'] = stock_name
        
        # 将数据添加到列表中
        all_stocks_data.append(df)
        print(f"成功获取{stock_name}的数据")
        
    except Exception as e:
        print(f"获取{stock_name}数据时出错: {str(e)}")

# 合并所有股票数据
if all_stocks_data:
    final_df = pd.concat(all_stocks_data, ignore_index=True)
    
    # 重新排序列
    columns_order = ['stock_name', 'ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']
    final_df = final_df[columns_order]
    
    # 将日期格式转换为更易读的格式
    final_df['trade_date'] = pd.to_datetime(final_df['trade_date'])
    
    # 按照trade_date从小到大排序
    final_df = final_df.sort_values(by='trade_date', ascending=True)
    
    # 再将trade_date转回字符串格式
    final_df['trade_date'] = final_df['trade_date'].dt.strftime('%Y-%m-%d')
    
    # 保存到Excel文件
    output_file = 'stock_history_data.xlsx'
    final_df.to_excel(output_file, index=False)
    print(f"\n数据已保存到 {output_file}")
else:
    print("没有获取到任何数据") 