"""
LightGBM特征工程与建模
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from lightgbm.sklearn import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

"""
数据探索（EDA可视化）:
    我们拿到数据后，需要对数据进行初步探索，可以通过可视化图表等方式观察数据分布情况。使用EDA技术了解预测值的分布、特征分析等工作。
    数据集中有部分较明显地存在缺省，因此我们根据实际情况进行相应的填充/特征构造/删除/不处理等操作，这个步骤有利于后面的特征工程。
"""
Train_data = pd.read_csv('06不同领域的AI算法/二手车价格预测/used_car_train_20200313.csv',
                         sep=' ')  # handle_used_car_train.csv
Test_data = pd.read_csv('06不同领域的AI算法/二手车价格预测/used_car_testB_20200421.csv', sep=' ')
 
# 合并方便后面的操作
df = pd.concat([Train_data, Test_data], ignore_index=True)
print(pd.concat([df.head(), df.tail()]))


# ======对价格分布情况的查看，如下左图，发现价格price是偏分布，所以进行了对数转换======
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
from datetime import datetime
 
 
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
LightGBM模型预测:使用的模型是LightGBM的回归算法，是基于梯度提升的决策树算法，训练很快，在机器学习中应用广泛。这里将数据进行五折交叉验证，模型对训练数据进行训练，最后给出预测结果。
"""
# 建立模型预测
# 划分训练数据和测试数据
df1 = df.copy()
test = df1[df1['price'].isnull()]
X_train = df1[df1['price'].notnull()].drop(['price', 'regDate', 'creatDate', 'SaleID', 'regionCode'], axis=1)
Y_train = df1[df1['price'].notnull()]['price']
X_test = df1[df1['price'].isnull()].drop(['price', 'regDate', 'creatDate', 'SaleID', 'regionCode'], axis=1)
print(pd.concat([X_train.head(), X_train.tail()]))
# 五折交叉检验
cols = list(X_train)
oof = np.zeros(X_train.shape[0])
sub = test[['SaleID']].copy()
sub['price'] = 0
feat_df = pd.DataFrame({'feat': cols, 'imp': 0})
skf = KFold(n_splits=4, shuffle=True, random_state=2020)
 
# 修复LGBM模型参数问题 - 第136-145行附近
clf = LGBMRegressor(
    learning_rate=0.05,  # 学习率，控制每次迭代更新模型参数的步长，较小的值可以使模型收敛更稳定，但可能需要更多的迭代次数
    n_estimators=1500,  # 迭代次数，即训练的弱学习器（决策树）的数量
    max_depth=7,  # 决策树的最大深度，限制树的层数，防止模型过拟合
    num_leaves=31,  # 一棵树上的最大叶子节点数，LightGBM使用leaf-wise生长策略，该参数影响模型复杂度
    min_child_weight=0.01,  # 子节点的最小权重和，用于防止过拟合，值越大越容易欠拟合
    min_data_in_leaf=30,  # 叶子节点的最小样本数，防止过拟合，避免生成过小的叶子节点
    bagging_fraction=0.8,  # 每次迭代时用于训练模型的样本比例，可减少过拟合，提高训练速度
    lambda_l2=2,  # L2正则化系数，用于控制模型复杂度，值越大正则化效果越强
    random_state=2020,  # 随机数种子，保证每次运行结果的可复现性
    metric='mae',  # 模型评估指标，这里使用平均绝对误差（Mean Absolute Error）
    early_stopping_rounds=300,  # 早停参数，如果在300次迭代内评估指标没有提升，则提前停止训练
    verbose=300  # 训练过程中输出信息的间隔，每300次迭代输出一次训练信息
)
 
mae = 0
for i, (trn_idx, val_idx) in enumerate(skf.split(X_train, Y_train)):
    print('--------------------- {} fold ---------------------'.format(i + 1))
    trn_x, trn_y = X_train.iloc[trn_idx].reset_index(drop=True), Y_train[trn_idx]
    val_x, val_y = X_train.iloc[val_idx].reset_index(drop=True), Y_train[val_idx]
    # 在第151行附近修改fit方法调用，移除verbose参数
    clf.fit(
        trn_x, trn_y,
        eval_set=[(val_x, val_y)],
        eval_metric='mae'
    )
 
    sub['price'] += np.expm1(clf.predict(X_test)) / skf.n_splits
    oof[val_idx] = clf.predict(val_x)
    print('val mae:', mean_absolute_error(np.expm1(val_y), np.expm1(oof[val_idx])))
    mae += mean_absolute_error(np.expm1(val_y), np.expm1(oof[val_idx])) / skf.n_splits
 
print('cv mae:', mae)

# 生成提交文件
sub.to_csv('fe_lightgbm_submit_result.csv', index=False)
print("预测结果已保存到 fe_lightgbm_submit_result.csv")
