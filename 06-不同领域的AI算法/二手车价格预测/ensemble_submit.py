"""
融合CatBoost和XGBoost模型预测结果
"""

import pandas as pd

# 读取两个模型的预测结果
catboost_df = pd.read_csv('fe_catboost_submit_result.csv')
xgb_df = pd.read_csv('fe_xgb_submit_result.csv')

# 按照SaleID对齐，防止顺序错乱
catboost_df = catboost_df.sort_values('SaleID').reset_index(drop=True)
xgb_df = xgb_df.sort_values('SaleID').reset_index(drop=True)

# 检查SaleID是否完全一致
assert (catboost_df['SaleID'] == xgb_df['SaleID']).all(), 'SaleID不一致，无法融合！'

# 融合权重
w_cat = 599 / (602 + 599)
w_xgb = 602 / (602 + 599)

# 计算融合后的price
ensemble_price = catboost_df['price'] * w_cat + xgb_df['price'] * w_xgb

# 保存融合结果
ensemble_df = pd.DataFrame({
    'SaleID': catboost_df['SaleID'],
    'price': ensemble_price
})
ensemble_df.to_csv('fe_ensemble_submit_result.csv', index=False)
print('融合结果已保存到 fe_ensemble_submit_result.csv') 