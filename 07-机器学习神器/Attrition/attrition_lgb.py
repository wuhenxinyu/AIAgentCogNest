import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)

# 数据探索
print('训练集 Attrition 分布：')
print(train['Attrition'].value_counts())
print('\n训练集 Attrition 比例：')
print(train['Attrition'].value_counts(normalize=True))

# 处理Attrition字段
train['Attrition'] = train['Attrition'].map(lambda x: 1 if x == 'Yes' else 0)

# 去掉没用的列
train = train.drop(['EmployeeNumber', 'StandardHours'], axis=1)
test = test.drop(['EmployeeNumber', 'StandardHours'], axis=1)

# 对于分类特征进行特征值编码
attr = ['Age', 'BusinessTravel', 'Department', 'Education', 'EducationField', 
        'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']
lbe_list = []
for feature in attr:
    lbe = LabelEncoder()
    train[feature] = lbe.fit_transform(train[feature])
    test[feature] = lbe.transform(test[feature])
    lbe_list.append(lbe)

import lightgbm as lgb
from sklearn.model_selection import train_test_split

# 模型参数
param = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.01,
    'max_depth': 15,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.9,
    'bagging_freq': 8,
    'lambda_l1': 0.6,
    'lambda_l2': 0,
    'is_unbalance': True  # 处理类别不平衡
}

# 数据集切分
X_train, X_valid, y_train, y_valid = train_test_split(
    train.drop('Attrition', axis=1), 
    train['Attrition'], 
    test_size=0.2, 
    random_state=42
)

# 创建数据集
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid)

# 训练模型
callbacks = [
    lgb.early_stopping(stopping_rounds=200),
    lgb.log_evaluation(period=25)
]

model = lgb.train(
    params=param,
    train_set=train_data,
    valid_sets=[train_data, valid_data],
    num_boost_round=10000,
    callbacks=callbacks,
    categorical_feature=attr
)

# 获取预测概率
proba = model.predict(test)

# 计算目标阈值
target_ratio = 0.16  # 目标比例
n_samples = len(test)
target_positive = int(n_samples * target_ratio)

# 根据概率排序，选择前 target_positive 个样本作为正类
threshold = np.sort(proba)[-target_positive]

# 使用调整后的阈值进行预测
predict = (proba >= threshold).astype(int)

# 输出结果统计
print('\n预测结果统计：')
print('预测的 Attrition 数量：', np.sum(predict))
print('预测的 Attrition 比例：', np.sum(predict) / len(predict))
print('使用的阈值：', threshold)

# 保存结果
test['Attrition'] = predict
test[['Attrition']].to_csv('submit_lgb.csv')
print('\n结果已保存到 submit_lgb.csv')
