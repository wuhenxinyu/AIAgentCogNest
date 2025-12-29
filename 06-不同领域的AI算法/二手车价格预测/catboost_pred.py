import pandas as pd
from catboost import CatBoostRegressor, Pool
import joblib

# 加载模型
model = CatBoostRegressor()
model.load_model('processed_data/fe_catboost_model.cbm')

# 加载测试集数据和SaleID
X_test = joblib.load('processed_data/fe_test_data.joblib')
test_ids = joblib.load('processed_data/fe_sale_ids.joblib')

# 指定分类特征
cat_features = [
    'model',
    'brand',
    'bodyType',
    'fuelType',
    'gearbox',
    'notRepairedDamage',
    'age_segment',
    'brand_model'
]

def predict_test_data(model, X_test, test_ids, cat_features):
    """
    预测测试集数据
    """
    print("\n正在预测测试集...")
    
    # 创建测试数据池
    test_pool = Pool(X_test, cat_features=cat_features)
    
    # 预测
    predictions = model.predict(test_pool)
    
    # 创建提交文件
    submit_data = pd.DataFrame({
        'SaleID': test_ids,
        'price': predictions
    })
    
    # 保存预测结果
    submit_data.to_csv('fe_catboost_submit_result1.csv', index=False)
    print("预测结果已保存到 fe_catboost_submit_result1.csv")

# 预测测试集
predict_test_data(model, X_test, test_ids, cat_features)

print("\n模型训练、评估和预测完成！")
