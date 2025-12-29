"""
二手车价格预测 - XGBoost建模（直接复用特征工程结果）
"""

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载特征工程后的数据
X_train = joblib.load('processed_data/fe_X_train.joblib')
X_val = joblib.load('processed_data/fe_X_val.joblib')
y_train = joblib.load('processed_data/fe_y_train.joblib')
y_val = joblib.load('processed_data/fe_y_val.joblib')
X_test = joblib.load('processed_data/fe_test_data.joblib')
test_ids = joblib.load('processed_data/fe_sale_ids.joblib')

# 2. XGBoost训练
print("开始训练XGBoost模型...")

# XGBoost不支持直接用category类型，需转为int
for col in X_train.select_dtypes(include='category').columns:
    X_train[col] = X_train[col].cat.codes
    X_val[col] = X_val[col].cat.codes
    X_test[col] = X_test[col].cat.codes

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test)

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'learning_rate': 0.01,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42,
    'nthread': -1
}

evals = [(dtrain, 'train'), (dval, 'val')]
model = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    evals=evals,
    early_stopping_rounds=20,
    verbose_eval=100
)

# 3. 验证集评估
y_pred_val = model.predict(dval)
mse = mean_squared_error(y_val, y_pred_val)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_val, y_pred_val)
r2 = r2_score(y_val, y_pred_val)

print("\n模型评估结果：")
print(f"均方根误差 (RMSE): {rmse:.2f}")
print(f"平均绝对误差 (MAE): {mae:.2f}")
print(f"R2分数: {r2:.4f}")

# 4. 特征重要性
importance = model.get_score(importance_type='weight')
importance_df = pd.DataFrame({
    'feature': list(importance.keys()),
    'importance': list(importance.values())
}).sort_values('importance', ascending=False)
importance_df.to_csv('fe_xgb_feature_importance.csv', index=False)

plt.figure(figsize=(14, 8))
sns.barplot(x='importance', y='feature', data=importance_df.head(20))
plt.title('XGBoost Top 20 特征重要性')
plt.tight_layout()
plt.savefig('fe_xgb_feature_importance.png')
plt.close()

# 5. 预测测试集并保存
y_pred_test = model.predict(dtest)
submit_data = pd.DataFrame({
    'SaleID': test_ids,
    'price': y_pred_test
})
submit_data.to_csv('fe_xgb_submit_result.csv', index=False)
print("预测结果已保存到 fe_xgb_submit_result.csv") 