import pandas as pd

# 定义客户标签和对应推荐话术
customer_tags = ['青年', '中年', '老年']
recommend_speech = [
    '尊敬的青年客户，推荐您关注我们的智能理财产品和基金定投计划，助力财富快速增长！',
    '尊敬的中年客户，建议您配置多元化理财产品，兼顾收益与稳健，规划家庭未来！',
    '尊敬的老年客户，推荐您选择稳健型理财和专属养老保险产品，保障晚年生活无忧！'
]

# 组装数据
speech_df = pd.DataFrame({
    '客户标签': customer_tags,
    '推荐话术': recommend_speech
})

# 写入Excel
speech_df.to_excel('客户话术.xlsx', index=False)
print('客户话术.xlsx 已生成！')

# 中文注释：
# 本脚本用于生成客户标签与推荐话术的Excel文件，便于后续针对不同客户群体精准营销。 