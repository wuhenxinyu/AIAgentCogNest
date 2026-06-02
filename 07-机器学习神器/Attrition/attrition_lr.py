import pandas as pd
from sklearn.feature_extraction import DictVectorizer

# 数据加载
train=pd.read_csv('train.csv',index_col=0)
test=pd.read_csv('test.csv',index_col=0)

# 数据探索
print(train['Attrition'].value_counts())

# 处理Attrition字段, 可以使用map 进行自定义，也可以使用LabelEncoder进行自动的标签编码
train['Attrition']=train['Attrition'].map(lambda x:1 if x=='Yes' else 0)
print(train['Attrition'].value_counts())

from sklearn.preprocessing import LabelEncoder
# 查看数据是否有空值
print(train.isnull().sum())
# 如果方差为0, 没有意义
print(train['StandardHours'].value_counts())

# 去掉没用的列 员工号码，标准工时（=80）
train = train.drop(['EmployeeNumber', 'StandardHours'], axis=1)
test = test.drop(['EmployeeNumber', 'StandardHours'], axis=1)
print(train.info())

# 对于分类特征进行特征值编码
attr=['BusinessTravel','Department','Education','EducationField','Gender','JobRole','MaritalStatus','Over18','OverTime']
lbe_list=[]
# 在这个数据集中，测试集出现的标签，在训练集中都出现过
# 一般还可以，将训练集和测试集统一起来，一起进行fit_transform
for feature in attr:
    # 标签编码： 如果有10个类别，会编码成0-9
    lbe=LabelEncoder()
    # fit_transform = 先fit 再 transform
    # fit就是指定 LabelEncoder的关系，transform是应用这种LabelEncoder的关系进行编码
    train[feature]=lbe.fit_transform(train[feature])
    # 对测试集的特征值 不需要进行fit
    # 如果对测试集进行了fit, 训练集和测试集的lbe标准 就不一样了
    test[feature]=lbe.transform(test[feature])
    lbe_list.append(lbe)
#print(train)
train.to_csv('train_label_encoder.csv')

# 建模环节，分类模型
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# 数据集进行切分，20%用于测试
X_train, X_valid, y_train, y_valid = train_test_split(train.drop('Attrition',axis=1), train['Attrition'], test_size=0.2, random_state=2025)

# 分类模型 二分类
# 为什么写random_state？如果不写random_state 每次运行的结果不同
model = LogisticRegression(
    max_iter=100,
    verbose=True,
    random_state=42,
    tol=1e-4,
    penalty='l2',  # 或 'l1'
    C=1.0,         # 正则化强度倒数，越小正则化越强
)
# 模型训练
model.fit(X_train, y_train)
# To DO 还可以使用验证集，提前了解模型的效果
# 二分类结果，0或1
predict = model.predict(test)
print('标签Label：')
print(predict)
# 二分类任务 有2个概率值，label=0的概率， label=1的概率
print('标签概率')
predict = model.predict_proba(test)[:, 1]
print(predict)
"""
test['Attrition']=predict

print(test['Attrition'])
test[['Attrition']].to_csv('submit_lr.csv')
print('submit_lr.csv saved')
# 转化为二分类输出
test['Attrition']=test['Attrition'].map(lambda x:1 if x>=0.5 else 0)
#test[['Attrition']].to_csv('submit_lr.csv')

#198     0.116689
#1229    0.160023

#198     0.260560
#1229    0.153033
"""