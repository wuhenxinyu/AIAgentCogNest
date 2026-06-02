import pandas as pd
train=pd.read_csv('train.csv',index_col=0)
test=pd.read_csv('test.csv',index_col=0)

#print(train['Attrition'].value_counts())
# 处理Attrition字段
train['Attrition']=train['Attrition'].map(lambda x:1 if x=='Yes' else 0)
from sklearn.preprocessing import LabelEncoder
# 查看数据是否有空值
#print(train.isna().sum())

# 去掉没用的列 员工号码，标准工时（=80）
train = train.drop(['EmployeeNumber', 'StandardHours'], axis=1)
test = test.drop(['EmployeeNumber', 'StandardHours'], axis=1)

# 对于分类特征进行特征值编码
attr=['Age','BusinessTravel','Department','Education','EducationField','Gender','JobRole','MaritalStatus','Over18','OverTime']
lbe_list=[]
for feature in attr:
    lbe=LabelEncoder()
    train[feature]=lbe.fit_transform(train[feature])
    test[feature]=lbe.transform(test[feature])
    lbe_list.append(lbe)
#train.to_csv('temp.csv')
#print(train)

import xgboost as xgb
from sklearn.model_selection import train_test_split

param = {'boosting_type':'gbdt',
                         'objective' : 'binary:logistic', #目标函数
                         'eval_metric' : 'auc', # 评价指标
                         'eta' : 0.01,
                         'max_depth' : 6,
                         'colsample_bytree':0.8,
                         'subsample': 0.9,
                         'subsample_freq': 8,
                         'alpha': 0.6,
                         'lambda': 0
        }
X_train, X_valid, y_train, y_valid = train_test_split(train.drop('Attrition',axis=1), train['Attrition'], test_size=0.2, random_state=42)

train_data = xgb.DMatrix(X_train, label=y_train)
valid_data = xgb.DMatrix(X_valid, label=y_valid)
test_data = xgb.DMatrix(test)

model = xgb.train(param, train_data, evals=[(train_data, 'train'), (valid_data, 'valid')], num_boost_round = 1000, early_stopping_rounds=10, verbose_eval=25)
predict = model.predict(test_data)
test['Attrition']=predict
print(predict)
# 转化为二分类输出
#test['Attrition']=test['Attrition'].map(lambda x:1 if x>=0.5 else 0)
test[['Attrition']].to_csv('submit_xgb.csv')
