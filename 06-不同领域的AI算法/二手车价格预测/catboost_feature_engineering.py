"""
高级特征工程与CatBoost建模
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from datetime import datetime

"""
数据探索（EDA可视化）:
    我们拿到数据后，需要对数据进行初步探索，可以通过可视化图表等方式观察数据分布情况。使用EDA技术了解预测值的分布、特征分析等工作。
    数据集中有部分较明显地存在缺省，因此我们根据实际情况进行相应的填充/特征构造/删除/不处理等操作，这个步骤有利于后面的特征工程。
"""
# 加载数据
Train_data = pd.read_csv('06不同领域的AI算法/二手车价格预测/used_car_train_20200313.csv',sep=' ')
Test_data = pd.read_csv('06不同领域的AI算法/二手车价格预测/used_car_testB_20200421.csv', sep=' ')
 
# 合并方便后面的操作
df = pd.concat([Train_data, Test_data], ignore_index=True)
print(pd.concat([df.head(), df.tail()]))


# ======对价格分布情况的查看，发现价格price是偏分布，所以进行了对数转换======
# 对'price'做对数变换
df['price'] = np.log1p(df['price'])

"""
预处理:接着上一步的实验结果，现在需要对数据进行初步地处理，获取有效的特征信息，更好地给目标预测结果提供正确的预测方向。
"""
# 用众数填充缺失值
df['fuelType'] = df['fuelType'].fillna(0)
df['gearbox'] = df['gearbox'].fillna(0)
df['bodyType'] = df['bodyType'].fillna(0)
df['model'] = df['model'].fillna(0)
 
# 处理异常值
df['power'] = df['power'].map(lambda x: 600 if x > 600 else x)  # 赛题限定power<=600
df['notRepairedDamage'] = df['notRepairedDamage'].astype('str').apply(lambda x: x if x != '-' else None).astype(
    'float32')
 
# 对可分类的连续特征进行分桶，kilometer是已经分桶了
bin = [i * 10 for i in range(31)]
df['power_bin'] = pd.cut(df['power'], bin, labels=False)
 
bin = [i * 10 for i in range(24)]
df['model_bin'] = pd.cut(df['model'], bin, labels=False)

"""
特征构造:经过数据分析，会发现数据存在数值型和类别型的属性，实际上它们之间存在许多联系，比如时间信息、行驶路程等数值型特征，对于价格的影响是很大的，所以，我们可以采用特征构造的方法，提取更有效的数据信息。
"""
# 特征工程
# 时间提取出年，月，日和使用时间


def date_process(x):
    year = int(str(x)[:4])
    month = int(str(x)[4:6])
    day = int(str(x)[6:8])
 
    if month < 1:
        month = 1
 
    date = datetime(year, month, day)
    return date
 

df['regDate'] = df['regDate'].apply(date_process)
df['creatDate'] = df['creatDate'].apply(date_process)
df['regDate_year'] = df['regDate'].dt.year
df['regDate_month'] = df['regDate'].dt.month
df['regDate_day'] = df['regDate'].dt.day
df['creatDate_year'] = df['creatDate'].dt.year
df['creatDate_month'] = df['creatDate'].dt.month
df['creatDate_day'] = df['creatDate'].dt.day
 
# 二手车使用天数
df['car_age_day'] = (df['creatDate'] - df['regDate']).dt.days
# 二手车使用年数
df['car_age_year'] = round(df['car_age_day'] / 365, 1)
 
# 行驶路程与功率统计
kk = ['kilometer', 'power']
t1 = Train_data.groupby(kk[0], as_index=False)[kk[1]].agg(
    {kk[0] + '_' + kk[1] + '_count': 'count', kk[0] + '_' + kk[1] + '_max': 'max',
     kk[0] + '_' + kk[1] + '_median': 'median',
     kk[0] + '_' + kk[1] + '_min': 'min', kk[0] + '_' + kk[1] + '_sum': 'sum', kk[0] + '_' + kk[1] + '_std': 'std',
     kk[0] + '_' + kk[1] + '_mean': 'mean'})
df = pd.merge(df, t1, on=kk[0], how='left')

"""
类别特征交叉:通过对14个匿名特征的相关性探索，发现匿名特征v_0,v_3,v_8,v_12与价格存在明显较大相关性。作为数值型特征，尝试进行了四则运算简单组合，交叉构造新的特征，作为模型训练特征信息，对汽车价格预测的准确性进一步提高了。
"""
# v_0,v_3,v_8,v_12与price的相关性很高，所以做四则运算简单组合，发现效果不错
num_cols = [0, 3, 8, 12]
for i in num_cols:
    for j in num_cols:
        df['new' + str(i) + '*' + str(j)] = df['v_' + str(i)] * df['v_' + str(j)]
 
for i in num_cols:
    for j in num_cols:
        df['new' + str(i) + '+' + str(j)] = df['v_' + str(i)] + df['v_' + str(j)]
 
for i in num_cols:
    for j in num_cols:
        df['new' + str(i) + '-' + str(j)] = df['v_' + str(i)] - df['v_' + str(j)]
 
for i in range(15):
    df['new' + str(i) + '*year'] = df['v_' + str(i)] * df['car_age_year']

"""
CatBoost模型预测:使用的模型是CatBoost的回归算法，是基于梯度提升的决策树算法，训练很快，在机器学习中应用广泛。这里将数据进行五折交叉验证，模型对训练数据进行训练，最后给出预测结果。
"""
# 建立模型预测
# 划分训练数据和测试数据
df1 = df.copy()
test = df1[df1['price'].isnull()]
X_train = df1[df1['price'].notnull()].drop(['price', 'regDate', 'creatDate', 'SaleID', 'regionCode'], axis=1)
Y_train = df1[df1['price'].notnull()]['price']
X_test = df1[df1['price'].isnull()].drop(['price', 'regDate', 'creatDate', 'SaleID', 'regionCode'], axis=1)
print(pd.concat([X_train.head(), X_train.tail()]))

# 处理分类特征
# 在CatBoost中，我们需要指定哪些特征是分类特征
cat_features = ['fuelType', 'gearbox', 'bodyType', 'brand', 'model', 'notRepairedDamage', 'power_bin', 'model_bin']

# 将分类特征转换为字符串类型以便CatBoost处理
for col in cat_features:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype('str')
        X_test[col] = X_test[col].astype('str')

# 五折交叉检验
cols = list(X_train)
oof = np.zeros(X_train.shape[0])
sub = test[['SaleID']].copy()
sub['price'] = 0
feat_df = pd.DataFrame({'feat': cols, 'imp': 0})
skf = KFold(n_splits=4, shuffle=True, random_state=2020)
 
# 设置CatBoost模型参数
# 初始化 CatBoost 回归模型实例，CatBoost 是基于梯度提升的决策树算法，适用于回归和分类任务
clf = CatBoostRegressor(
    # 学习率，控制每次迭代时模型参数更新的步长。略微提高学习率（当前设置为0.05）可以加快模型收敛速度，但过大可能导致模型无法收敛。
    learning_rate=0.05,
    # 迭代次数，即模型训练过程中构建决策树的数量。增加迭代次数（当前设置为13000）能让模型有更多学习机会，提升模型性能，但也会增加训练时间。
    iterations=13000,
    # 每棵决策树的最大深度，限制了树的生长层次。增加深度（当前设置为8）可以捕获更复杂的特征关系，但可能导致过拟合。
    depth=7,
    # L2 正则化系数，用于控制模型的复杂度。降低正则化强度（当前设置为2）可以减少欠拟合的风险，但可能增加过拟合的可能性。
    l2_leaf_reg=3,
    # 自助采样类型，决定了在构建每棵树时如何对样本进行采样。'Bernoulli' 是一种随机采样方式，以一定概率对样本进行采样。
    bootstrap_type='Bayesian',
    # 随机种子，保证每次运行代码时模型的随机过程结果一致，使得实验结果可复现。
    random_state=2020,
    # 损失函数，用于衡量模型预测值与真实值之间的差异。'RMSE'（均方根误差）是回归问题中常用的损失函数。
    loss_function='MAE',
    # 评估指标，用于在训练过程中评估模型的性能。'MAE'（平均绝对误差）可以直观反映预测值与真实值的平均误差。
    eval_metric='MAE',
    # 任务运行的设备类型，'GPU' 表示使用图形处理器进行模型训练，可显著提高训练速度。
    task_type='CPU',
    # 当使用GPU时，指定使用的GPU设备索引，0表示第一块GPU。如果有多个GPU，可以指定多个索引，如'0:1:2'
    # devices='0',
    # GPU训练时的内存消耗上限，以MB为单位。设置合适的值可以避免GPU内存溢出。
    # gpu_ram_part=0.9,
    # 线程数，设置为 -1 表示使用所有可用的 CPU 线程，主要用于数据预处理阶段。
    thread_count=-1,
    # 提前停止回合数，当连续指定次数（当前设置为200）的迭代中评估指标没有提升时，停止训练，避免过拟合。
    early_stopping_rounds=300,
    # 训练过程中输出日志的频率，每训练指定次数（当前设置为300）的迭代后输出一次训练信息。
    verbose=300
)
 
mae = 0
for i, (trn_idx, val_idx) in enumerate(skf.split(X_train, Y_train)):
    print('--------------------- {} fold ---------------------'.format(i + 1))
    trn_x, trn_y = X_train.iloc[trn_idx].reset_index(drop=True), Y_train[trn_idx]
    val_x, val_y = X_train.iloc[val_idx].reset_index(drop=True), Y_train[val_idx]
    
    # 训练模型并添加样本权重以解决价格预测问题
    clf.fit(
        trn_x, trn_y,
        eval_set=[(val_x, val_y)],
        cat_features=cat_features,
        use_best_model=True,
        # 添加样本权重，对高价样本给予更多关注
        sample_weight=np.exp(trn_y)
    )

    # 预测测试集和验证集
    sub['price'] += np.expm1(clf.predict(X_test)) / skf.n_splits
    oof[val_idx] = clf.predict(val_x)
    print('val mae:', mean_absolute_error(np.expm1(val_y), np.expm1(oof[val_idx])))
    mae += mean_absolute_error(np.expm1(val_y), np.expm1(oof[val_idx])) / skf.n_splits
 
print('cv mae:', mae)

# 生成提交文件
sub.to_csv('fe_catboost_submit_result.csv', index=False)
print("预测结果已保存到 fe_catboost_submit_result.csv")

# 特征重要性分析
feature_importance = clf.get_feature_importance()
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("Top 10 重要特征:")
print(importance_df.head(10))

# 保存特征重要性
importance_df.to_csv('catboost_feature_importance.csv', index=False)
print("特征重要性已保存到 catboost_feature_importance.csv")