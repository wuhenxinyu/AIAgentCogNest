import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer

train=pd.read_csv('train.csv',index_col=0)
test=pd.read_csv('test.csv',index_col=0)
#print(train['Attrition'].value_counts())
# 处理Attrition字段
train['Attrition']=train['Attrition'].map(lambda x:1 if x=='Yes' else 0)

# 查看数据是否有空值
#print(train.isna().sum())

# 去掉没用的列 员工号码，标准工时（=80）
train = train.drop(['EmployeeNumber', 'StandardHours'], axis=1)
test = test.drop(['EmployeeNumber', 'StandardHours'], axis=1)

# 对于分类特征进行特征值编码
attr=['Age','BusinessTravel','Department','Education','EducationField','Gender','JobRole','MaritalStatus','Over18','OverTime']

import xgboost as xgb
from sklearn.model_selection import train_test_split

param = {'boosting_type':'gbdt',
                         'objective' : 'binary:logistic', #
                         'eval_metric' : 'auc',
                         'eta' : 0.01,
                         'max_depth' : 15,
                         'colsample_bytree':0.8,
                         'subsample': 0.9,
                         'subsample_freq': 8,
                         'alpha': 0.6,
                         'lambda': 0,
        }
X_train, X_valid, y_train, y_valid = train_test_split(train.drop('Attrition',axis=1), train['Attrition'], test_size=0.2, random_state=42)
# 使用DictVectorizer对离散值进行编码
dvec = DictVectorizer(sparse=False)
X_train = dvec.fit_transform(X_train.to_dict(orient='record'))
X_valid = dvec.transform(X_valid.to_dict(orient='record'))
temp_test = dvec.transform(test.to_dict(orient='record'))

# 进行DMatrix转换
train_data = xgb.DMatrix(X_train, label=y_train)
valid_data = xgb.DMatrix(X_valid, label=y_valid)
test_data = xgb.DMatrix(temp_test)

model = xgb.train(param, train_data, evals=[(train_data, 'train'), (valid_data, 'valid')], num_boost_round = 10000, early_stopping_rounds=200, verbose_eval=25)
predict = model.predict(test_data)
test['Attrition']=predict
print(predict)
# 转化为二分类输出
#test['Attrition']=test['Attrition'].map(lambda x:1 if x>=0.5 else 0)
test[['Attrition']].to_csv('submit_xgb.csv')
