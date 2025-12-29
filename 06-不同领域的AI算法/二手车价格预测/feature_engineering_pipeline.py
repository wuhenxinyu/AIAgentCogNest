import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from catboost import CatBoostRegressor, Pool
import joblib
import datetime
import warnings
import os  # 添加os模块导入
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class UsedCarFeatureEngineering:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.processed_data = None
        self.processed_test_data = None
        self.selected_features = None
        self.label_encoders = {}
        self.model = None
        self.y = None
        self.train_ids = None
        self.test_ids = None
        self.cat_features = None
        self.train_idx = None
        self.test_idx = None
        
    def load_data(self, train_path='used_car_train_20200313.csv', 
                 test_path='used_car_testB_20200421.csv'):
        """加载训练集和测试集数据"""
        print("正在加载数据...")
        try:
            # 获取当前脚本所在目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # 构建完整的文件路径
            full_train_path = os.path.join(script_dir, train_path)
            full_test_path = os.path.join(script_dir, test_path)
            
            # 首先尝试使用空格分隔符（这是正确的分隔方式）
            print(f"使用空格分隔符读取数据...\n训练集路径: {full_train_path}\n测试集路径: {full_test_path}")
            self.train_data = pd.read_csv(full_train_path, sep=' ')
            self.test_data = pd.read_csv(full_test_path, sep=' ')
            
            print(f"训练集形状: {self.train_data.shape}")
            print(f"测试集形状: {self.test_data.shape}")
            print(f"训练集列名: {list(self.train_data.columns[:5])}...")  # 显示前5个列名
        except Exception as e:
            print(f"数据加载出错: {e}")
        return self
        
    def data_overview(self):
        """数据概览，查看缺失值和基本统计信息"""
        print("\n=== 数据概览 ===")
        # 检查训练集缺失值
        missing_train = self.train_data.isnull().sum()
        print("\n训练集缺失值统计:")
        print(missing_train[missing_train > 0])
        
        # 检查测试集缺失值
        if self.test_data is not None:
            missing_test = self.test_data.isnull().sum()
            print("\n测试集缺失值统计:")
            print(missing_test[missing_test > 0])
        
        # 查看数据类型
        print("\n数据类型:")
        print(self.train_data.dtypes)
        return self
        
    def preprocess_data(self):
        """数据预处理，合并训练集和测试集"""
        print("\n=== 开始数据预处理 ===")
        
        # 合并训练集和测试集进行特征工程
        train_data = self.train_data.copy()
        test_data = self.test_data.copy()
        
        train_data['source'] = 'train'
        test_data['source'] = 'test'
        self.processed_data = pd.concat([train_data, test_data], ignore_index=True)
        
        # 保存SaleID
        self.train_ids = train_data['SaleID']
        self.test_ids = test_data['SaleID']
        
        # 从训练集获取y值
        self.y = train_data['price']
        
        # 找回训练集和测试集的索引
        self.train_idx = self.processed_data[self.processed_data['source'] == 'train'].index
        self.test_idx = self.processed_data[self.processed_data['source'] == 'test'].index
        
        return self
        
    def handle_missing_values(self):
        """处理缺失值"""
        print("\n=== 处理缺失值 ===")
        
        # 将'-'转换为NaN
        if 'notRepairedDamage' in self.processed_data.columns:
            self.processed_data['notRepairedDamage'] = self.processed_data['notRepairedDamage'].replace('-', np.nan)
        
        # 对于分类特征，使用众数填充
        categorical_cols = ['bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']
        for col in categorical_cols:
            if col in self.processed_data.columns:
                mode_value = self.processed_data[col].mode()[0]
                self.processed_data[col].fillna(mode_value, inplace=True)
                print(f"用众数 {mode_value} 填充 {col} 的缺失值")
        
        # 检查并处理其他可能包含'-'的列
        for col in self.processed_data.columns:
            # 检查列中是否有'-'字符
            if self.processed_data[col].dtype == 'object':
                try:
                    # 尝试将列转换为数值类型，处理'-'字符
                    self.processed_data[col] = pd.to_numeric(self.processed_data[col].replace('-', np.nan))
                except ValueError:
                    # 如果无法转换为数值，保持原样
                    pass
        
        return self
        
    def create_time_features(self):
        """创建时间特征"""
        print("\n=== 创建时间特征 ===")
        
        # 转换日期格式
        self.processed_data['regDate'] = pd.to_datetime(self.processed_data['regDate'], format='%Y%m%d', errors='coerce')
        self.processed_data['creatDate'] = pd.to_datetime(self.processed_data['creatDate'], format='%Y%m%d', errors='coerce')
        
        # 处理无效日期
        self.processed_data.loc[self.processed_data['regDate'].isnull(), 'regDate'] = pd.to_datetime('20160101', format='%Y%m%d')
        self.processed_data.loc[self.processed_data['creatDate'].isnull(), 'creatDate'] = pd.to_datetime('20160101', format='%Y%m%d')
        
        # 车辆年龄（天数）
        self.processed_data['vehicle_age_days'] = (self.processed_data['creatDate'] - self.processed_data['regDate']).dt.days
        
        # 修复异常值
        self.processed_data.loc[self.processed_data['vehicle_age_days'] < 0, 'vehicle_age_days'] = 0
        
        # 车辆年龄（年）
        self.processed_data['vehicle_age_years'] = self.processed_data['vehicle_age_days'] / 365
        
        # 注册年份和月份
        self.processed_data['reg_year'] = self.processed_data['regDate'].dt.year
        self.processed_data['reg_month'] = self.processed_data['regDate'].dt.month
        self.processed_data['reg_day'] = self.processed_data['regDate'].dt.day
        
        # 创建年份和月份
        self.processed_data['creat_year'] = self.processed_data['creatDate'].dt.year
        self.processed_data['creat_month'] = self.processed_data['creatDate'].dt.month
        self.processed_data['creat_day'] = self.processed_data['creatDate'].dt.day
        
        # 是否为新车（使用年限<1年）
        self.processed_data['is_new_car'] = (self.processed_data['vehicle_age_years'] < 1).astype(int)
        
        # 季节特征
        self.processed_data['reg_season'] = self.processed_data['reg_month'].apply(lambda x: (x%12 + 3)//3)
        self.processed_data['creat_season'] = self.processed_data['creat_month'].apply(lambda x: (x%12 + 3)//3)
        
        # 每年行驶的公里数
        self.processed_data['km_per_year'] = self.processed_data['kilometer'] / (self.processed_data['vehicle_age_years'] + 0.1)
        
        # 车龄分段
        self.processed_data['age_segment'] = pd.cut(self.processed_data['vehicle_age_years'], 
                                    bins=[-0.01, 1, 3, 5, 10, 100], 
                                    labels=['0-1年', '1-3年', '3-5年', '5-10年', '10年以上'])
        
        return self
        
    def create_car_features(self):
        """创建车辆特征"""
        print("\n=== 创建车辆特征 ===")
        
        # 缺失值处理
        numerical_features = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14']
        for feature in numerical_features:
            if feature in self.processed_data.columns:
                # 标记缺失值
                self.processed_data[f'{feature}_missing'] = self.processed_data[feature].isnull().astype(int)
                # 填充缺失值
                self.processed_data[feature] = self.processed_data[feature].fillna(self.processed_data[feature].median())
        
        # 将model转换为数值型特征
        if 'model' in self.processed_data.columns:
            self.processed_data['model_num'] = self.processed_data['model'].astype('category').cat.codes
        
        # 品牌与车型组合
        if 'brand' in self.processed_data.columns and 'model' in self.processed_data.columns:
            self.processed_data['brand_model'] = self.processed_data['brand'].astype(str) + '_' + self.processed_data['model'].astype(str)
        
        # 相对年份特征
        current_year = datetime.datetime.now().year
        if 'reg_year' in self.processed_data.columns:
            self.processed_data['car_age_from_now'] = current_year - self.processed_data['reg_year']
        
        # 处理异常值
        numerical_cols = ['power', 'kilometer', 'v_0']
        for col in numerical_cols:
            if col in self.processed_data.columns:
                Q1 = self.processed_data[col].quantile(0.05)
                Q3 = self.processed_data[col].quantile(0.95)
                IQR = Q3 - Q1
                self.processed_data[f'{col}_outlier'] = ((self.processed_data[col] < (Q1 - 1.5 * IQR)) | (self.processed_data[col] > (Q3 + 1.5 * IQR))).astype(int)
                self.processed_data[col] = self.processed_data[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        
        return self
        
    def create_statistical_features(self):
        """创建统计特征"""
        print("\n=== 创建统计特征 ===")
        
        # 仅使用训练集数据创建统计特征
        if self.train_idx is not None:
            train_data = self.processed_data.iloc[self.train_idx].reset_index(drop=True)
            
            # 品牌级别统计
            if 'brand' in self.processed_data.columns and 'price' in train_data.columns:
                brand_stats = train_data.groupby('brand').agg(
                    brand_price_mean=('price', 'mean'),
                    brand_price_median=('price', 'median'),
                    brand_price_std=('price', 'std'),
                    brand_price_count=('price', 'count')
                ).reset_index()
                
                # 合并统计特征
                self.processed_data = self.processed_data.merge(brand_stats, on='brand', how='left')
                
                # 相对价格特征（相对于平均价格）
                if 'brand_price_mean' in self.processed_data.columns:
                    self.processed_data['brand_price_ratio'] = self.processed_data['brand_price_mean'] / self.processed_data['brand_price_mean'].mean()
        
        return self
        
    def encode_categorical_features(self):
        """编码分类特征"""
        print("\n=== 编码分类特征 ===")
        
        # 目标编码的替代方案 - 频率编码
        categorical_cols = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']
        
        for col in categorical_cols:
            if col in self.processed_data.columns:
                # 填充缺失值
                self.processed_data[col] = self.processed_data[col].fillna('未知')
                
                # 频率编码
                freq_encoding = self.processed_data.groupby(col).size() / len(self.processed_data)
                self.processed_data[f'{col}_freq'] = self.processed_data[col].map(freq_encoding)
        
        # 将分类变量转换为CatBoost可以识别的格式
        self.cat_features = []
        for col in categorical_cols:
            if col in self.processed_data.columns:
                self.processed_data[col] = self.processed_data[col].astype('str')
                self.cat_features.append(col)
        
        return self
        
    def explore_anonymous_features(self):
        """挖掘匿名特征"""
        print("\n=== 挖掘匿名特征 ===")
        # 相关性分析，筛选与price高度相关的匿名特征
        anon_cols = [f'v_{i}' for i in range(15) if f'v_{i}' in self.processed_data.columns]
        
        # 特征交互，创建匿名特征的组合
        if 'v_0' in self.processed_data.columns and 'v_3' in self.processed_data.columns:
            # 确保两列都是数值类型
            if (pd.api.types.is_numeric_dtype(self.processed_data['v_0']) and 
                pd.api.types.is_numeric_dtype(self.processed_data['v_3'])):
                self.processed_data['v_0_v_3_interaction'] = self.processed_data['v_0'] * self.processed_data['v_3']
        if 'v_1' in self.processed_data.columns and 'v_2' in self.processed_data.columns:
            # 确保两列都是数值类型
            if (pd.api.types.is_numeric_dtype(self.processed_data['v_1']) and 
                pd.api.types.is_numeric_dtype(self.processed_data['v_2'])):
                self.processed_data['v_1_v_2_ratio'] = self.processed_data['v_1'] / (self.processed_data['v_2'] + 1e-5)  # 添加小值避免除零
        
        print("创建匿名特征交互项完成")
        return self
        
    def feature_selection(self):
        """特征选择和最终数据准备"""
        print("\n=== 特征选择和最终数据准备 ===")
        
        # 删除不再需要的列
        drop_cols = ['regDate', 'creatDate', 'price', 'SaleID', 'name', 'offerType', 'seller', 'source']
        self.processed_data = self.processed_data.drop(drop_cols, axis=1, errors='ignore')
        
        # 确保所有分类特征都被正确标记
        # 添加age_segment到分类特征列表中
        if 'age_segment' not in self.cat_features and 'age_segment' in self.processed_data.columns:
            self.cat_features.append('age_segment')
        
        # 确保brand_model也被标记为分类特征
        if 'brand_model' not in self.cat_features and 'brand_model' in self.processed_data.columns:
            self.cat_features.append('brand_model')
        
        # 转换分类特征
        for col in self.cat_features:
            if col in self.processed_data.columns:
                self.processed_data[col] = self.processed_data[col].astype('category')
        
        return self
        
    def split_data(self):
        """分离训练集和测试集，并划分训练集和验证集"""
        print("\n=== 分离训练集和测试集 ===")
        
        if self.train_idx is not None and self.test_idx is not None:
            # 分离训练集和测试集
            X_train_full = self.processed_data.iloc[self.train_idx].reset_index(drop=True)
            X_test = self.processed_data.iloc[self.test_idx].reset_index(drop=True)
            
            # 划分训练集和验证集
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, self.y, test_size=0.1, random_state=42
            )
            
            return X_train, X_val, y_train, y_val, X_test
        
        return None, None, None, None, None
        
    def train_catboost_model(self, X_train, X_val, y_train, y_val):
        """训练CatBoost模型"""
        print("\n=== 开始训练CatBoost模型 ===")
        
        # 创建数据池
        train_pool = Pool(X_train, y_train, cat_features=self.cat_features)
        val_pool = Pool(X_val, y_val, cat_features=self.cat_features)
        
        # 设置模型参数
        params = {
            'iterations': 3000,
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 3,
            'bootstrap_type': 'Bayesian',
            'random_seed': 42,
            'od_type': 'Iter',
            'od_wait': 100,
            'verbose': 100,
            'loss_function': 'MAE',
            'eval_metric': 'MAE',
            'task_type': 'CPU',
            'thread_count': -1
        }
        
        # 创建模型
        self.model = CatBoostRegressor(**params)
        
        # 训练模型
        self.model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
            plot=True
        )
        
        # 保存模型
        model_path = 'processed_data/fe_catboost_model.cbm'
        self.model.save_model(model_path)
        print(f"模型已保存到 {model_path}")
        
        return self.model
        
    def evaluate_model(self, model, X_val, y_val):
        """评估模型性能"""
        # 创建验证数据池
        val_pool = Pool(X_val, cat_features=self.cat_features)
        
        # 预测
        y_pred = model.predict(val_pool)
        
        # 计算评估指标
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        print("\n模型评估结果：")
        print(f"均方根误差 (RMSE): {rmse:.2f}")
        print(f"平均绝对误差 (MAE): {mae:.2f}")
        print(f"R2分数: {r2:.4f}")
        
        # 绘制预测值与实际值的对比图
        plt.figure(figsize=(10, 6))
        plt.scatter(y_val, y_pred, alpha=0.5)
        plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
        plt.xlabel('实际价格')
        plt.ylabel('预测价格')
        plt.title('CatBoost预测价格 vs 实际价格')
        plt.tight_layout()
        plt.savefig('fe_catboost_prediction_vs_actual.png')
        plt.close()
        
        return rmse, mae, r2
        
    def plot_feature_importance(self, model, X_train):
        """绘制特征重要性图"""
        # 获取特征重要性
        feature_importance = model.get_feature_importance()
        feature_names = X_train.columns
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # 保存特征重要性到CSV
        importance_df.to_csv('fe_catboost_feature_importance.csv', index=False)
        
        # 绘制特征重要性图
        plt.figure(figsize=(14, 8))
        sns.barplot(x='importance', y='feature', data=importance_df.head(20))
        plt.title('CatBoost Top 20 特征重要性')
        plt.tight_layout()
        plt.savefig('fe_catboost_feature_importance.png')
        plt.close()
        
        return importance_df
        
    def predict_test_data(self, model, X_test):
        """预测测试集数据"""
        print("\n=== 正在预测测试集 ===")
        
        # 创建测试数据池
        test_pool = Pool(X_test, cat_features=self.cat_features)
        
        # 预测
        predictions = model.predict(test_pool)
        
        # 创建提交文件
        submit_data = pd.DataFrame({
            'SaleID': self.test_ids,
            'price': predictions
        })
        
        # 保存预测结果
        submit_data.to_csv('fe_catboost_submit_result.csv', index=False)
        print("预测结果已保存到 fe_catboost_submit_result.csv")
        
        return submit_data
        
    def save_processed_data(self, output_path='processed_data.csv'):
        """保存处理后的数据"""
        print("\n=== 保存处理后的数据 ===")
        
        # 确保processed_data目录存在
        import os
        if not os.path.exists('processed_data'):
            os.makedirs('processed_data')
        
        # 分离训练集和测试集
        X_train, X_val, y_train, y_val, X_test = self.split_data()
        
        if X_train is not None:
            # 保存处理后的数据
            joblib.dump(X_train, 'processed_data/fe_X_train.joblib')
            joblib.dump(X_val, 'processed_data/fe_X_val.joblib')
            joblib.dump(y_train, 'processed_data/fe_y_train.joblib')
            joblib.dump(y_val, 'processed_data/fe_y_val.joblib')
            joblib.dump(X_test, 'processed_data/fe_test_data.joblib')
            joblib.dump(self.test_ids, 'processed_data/fe_sale_ids.joblib')
            joblib.dump(self.cat_features, 'processed_data/fe_cat_features.joblib')
            
            print("预处理后的数据已保存")
        
        return self
        
    def run_pipeline(self):
        """运行完整的特征工程流水线"""
        print("\n=== 开始特征工程流水线 ===")
        
        # 先加载数据
        self.load_data()
        self.data_overview()
        self.preprocess_data()
        self.handle_missing_values()
        self.create_time_features()
        self.create_car_features()
        self.create_statistical_features()
        self.encode_categorical_features()
        self.explore_anonymous_features()
        self.feature_selection()
        self.save_processed_data()
        
        # 分离数据并训练模型
        X_train, X_val, y_train, y_val, X_test = self.split_data()
        if X_train is not None:
            model = self.train_catboost_model(X_train, X_val, y_train, y_val)
            self.evaluate_model(model, X_val, y_val)
            importance_df = self.plot_feature_importance(model, X_train)
            self.predict_test_data(model, X_test)
            
            print("\n=== 模型训练、评估和预测完成 ===")
            print(f"Top 10 重要特征:\n{importance_df.head(10)}")
        
        print("\n=== 特征工程流水线完成 ===")
        return self

# 如果直接运行脚本，则执行特征工程流水线
if __name__ == "__main__":
    print("=== 二手车价格预测 - 特征工程和预测 ===")
    fe = UsedCarFeatureEngineering()
    fe.run_pipeline()