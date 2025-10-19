"""
使用LightGBM(树模型)预测客户未来3个月资产是否能提升至100万+
LightGBM的优点:
1. 高效,训练速度快,尤其在处理大规模数据集时
2. 内存占用低,对大规模数据集也能保持较好的性能
3. 支持并行计算,利用多核CPU优势
4. 对缺失值有良好的处理能力
5. 具有较高的预测准确性

roc曲线说明:
1. ROC曲线是用于评估分类模型性能的重要工具
2. 它展示了模型在不同阈值下的真阳性率（TPR）和假阳性率（FPR）
3. 曲线下的面积（AUC）越大，模型的分类能力越强
4. 一般来说，AUC值越接近1，模型的分类效果越好
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve

# 设置matplotlib支持中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 忽略警告信息
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """加载并预处理客户数据"""
    # 读取客户基础信息表
    try:
        df_base = pd.read_csv('28-AI运营助手/03预测高价值客户(分类算法)/customer_base.csv', encoding='utf-8')
    except UnicodeDecodeError:
        df_base = pd.read_csv('28-AI运营助手/03预测高价值客户(分类算法)/customer_base.csv', encoding='gbk')
        
    # 读取客户行为资产表
    try:
        df_behavior = pd.read_csv('28-AI运营助手/03预测高价值客户(分类算法)/customer_behavior_assets.csv', encoding='utf-8')
    except UnicodeDecodeError:
        df_behavior = pd.read_csv('28-AI运营助手/03预测高价值客户(分类算法)/customer_behavior_assets.csv', encoding='gbk')
    
    # 对每个客户，取最新的行为数据（按stat_month排序，取最新的一条）
    df_behavior_latest = df_behavior.sort_values(['customer_id', 'stat_month'], ascending=[True, False])
    df_behavior_latest = df_behavior_latest.drop_duplicates('customer_id', keep='first')
    
    # 合并两个表
    df_merged = pd.merge(df_base, df_behavior_latest, on='customer_id', how='inner')
    
    return df_merged

def prepare_features_and_labels(df):
    """准备模型所需的特征和标签"""
    # 选择业务相关的特征
    features = [
        'age',                    # 年龄
        'monthly_income',         # 月收入
        'total_assets',          # 当前总资产
        'product_count',         # 持有产品数量
        'app_login_count',       # 手机银行登录次数
        'financial_repurchase_count',  # 理财复购次数
        'investment_monthly_count',    # 月均投资次数
        'credit_card_monthly_expense', # 信用卡月消费额
        'app_financial_view_time',     # 理财浏览时长
        'app_product_compare_count'    # 产品比较次数
    ]
    
    # 处理分类变量
    le_gender = LabelEncoder()
    df['gender_encoded'] = le_gender.fit_transform(df['gender'])
    
    le_marriage = LabelEncoder()
    df['marriage_status_encoded'] = le_marriage.fit_transform(df['marriage_status'])
    
    le_city = LabelEncoder()
    df['city_level_encoded'] = le_city.fit_transform(df['city_level'])
    
    # 添加新的特征
    features.extend(['gender_encoded', 'marriage_status_encoded', 'city_level_encoded'])
    
    # 确保所有特征都存在
    existing_features = [f for f in features if f in df.columns]
    print(f"使用特征: {existing_features}")
    
    # 计算资产增长率和月收入对总资产的比例（这些指标会影响未来资产增长）
    df['income_to_asset_ratio'] = df['monthly_income'] / (df['total_assets'] + 1)  # 避免除零
    df['activity_score'] = (df['app_login_count'] + 1) * (df['investment_monthly_count'] + 1) * (df['financial_repurchase_count'] + 1)
    
    # 添加这些新特征
    df['income_to_asset_ratio'] = df['income_to_asset_ratio']
    df['activity_score'] = df['activity_score']
    existing_features.extend(['income_to_asset_ratio', 'activity_score'])
    
    # 构建标签：未来3个月资产是否达到100万+（基于业务逻辑构建）
    # 假设客户未来资产增长与当前资产、月收入、活跃度等因素正相关
    weight_current_assets = 0.7
    weight_monthly_income = 0.3
    weight_activity = 0.1
    weight_income_to_asset_ratio = 0.1
    
    # 计算综合增长潜力分数
    potential_score = (
        (df['total_assets'] / df['total_assets'].max()) * weight_current_assets +
        (df['monthly_income'] / df['monthly_income'].max()) * weight_monthly_income +
        (df['activity_score'] / df['activity_score'].max()) * weight_activity +
        (df['income_to_asset_ratio'] / df['income_to_asset_ratio'].max()) * weight_income_to_asset_ratio
    )
    
    # 设定阈值（前5%的客户为高价值客户）
    threshold = np.percentile(potential_score, 95)
    
    # 构建标签：未来3个月资产是否达到100万+
    df['label'] = (potential_score >= threshold).astype(int)
    
    print(f"标签分布: {df['label'].value_counts()}")
    
    # 特征值可能有缺失，用中位数填充
    X = df[existing_features].fillna(df[existing_features].median())
    y = df['label']
    
    return X, y, existing_features

def train_lightgbm_model(X, y):
    """训练LightGBM模型"""
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 创建LightGBM数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # 设置参数
    params = {
        'objective': 'binary',  # 二分类任务
        'metric': 'binary_logloss,auc',  # 评估指标
        'boosting_type': 'gbdt',  # 提升类型
        'num_leaves': 31,  # 叶子数
        'learning_rate': 0.05,  # 学习率
        'feature_fraction': 0.9,  # 特征采样比例
        'bagging_fraction': 0.8,  # 数据采样比例
        'bagging_freq': 5,  # bagging频率
        'verbose': -1,  # 静默模式
        'random_state': 42
    }
    
    # 训练模型
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]  # 早停和日志
    )
    
    # 预测
    y_pred = model.predict(X_test)
    y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]
    
    return model, X_train, X_test, y_train, y_test, y_pred_binary, y_pred

def print_feature_importance(model, features):
    """打印特征重要性排序"""
    # 获取特征重要性
    importance = model.feature_importance(importance_type='gain')  # 使用增益作为重要性指标
    
    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    print("\n=== 特征重要性排序（按增益值）===")
    print(importance_df.to_string(index=False))
    
    return importance_df

def visualize_feature_importance(model, features):
    """生成特征重要性可视化图片"""
    # 获取特征重要性
    importance = model.feature_importance(importance_type='gain')
    
    # 创建特征重要性DataFrame并排序
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values(by='Importance', ascending=True)  # 为了可视化颠倒顺序
    
    # 创建可视化图表
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importance_df)), importance_df['Importance'])
    plt.yticks(range(len(importance_df)), importance_df['Feature'])
    plt.xlabel('重要性（增益值）')
    plt.title('LightGBM特征重要性排序')
    plt.gca().invert_yaxis()  # 使最重要的特征在顶部
    
    # 保存图片
    plt.tight_layout()
    plt.savefig('lightgbm_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return importance_df

def evaluate_model(y_test, y_pred, y_pred_proba):
    """评估模型性能"""
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    
    # 计算AUC值
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # 生成分类报告
    report = classification_report(y_test, y_pred, digits=4)
    
    # 打印评估结果
    print('\n=== 模型评估结果 ===')
    print(f'准确率: {accuracy:.4f}')
    print(f'AUC值: {auc:.4f}')
    print('分类报告:')
    print(report)

def predict_new_customers(model, features, X):
    """对新客户进行预测"""
    print('\n=== 对部分客户进行预测 ===')
    
    # 随机选择部分客户进行预测演示
    sample_indices = np.random.choice(X.index, size=min(10, len(X)), replace=False)
    sample_X = X.loc[sample_indices]
    sample_customers = pd.DataFrame(sample_X)
    
    # 预测概率和类别
    sample_customers['预测概率'] = model.predict(sample_X)
    sample_customers['预测类别'] = [1 if pred > 0.5 else 0 for pred in sample_customers['预测概率']]
    
    # 获取原始数据以显示客户ID和当前资产
    try:
        df_original = load_and_preprocess_data()
        sample_customers['customer_id'] = df_original.loc[sample_indices, 'customer_id'].values
        sample_customers['total_assets'] = df_original.loc[sample_indices, 'total_assets'].values
    except:
        sample_customers['customer_id'] = sample_indices
        sample_customers['total_assets'] = sample_X['total_assets'].values if 'total_assets' in sample_X.columns else np.nan
    
    # 显示客户ID、当前资产、预测概率和预测类别
    result_cols = ['customer_id', 'total_assets', '预测概率', '预测类别']
    print(sample_customers[result_cols].to_string(index=False))

def visualize_roc_curve(y_test, y_pred_proba):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (False Positive Rate)')
    plt.ylabel('真正率 (True Positive Rate)')
    plt.title('LightGBM模型ROC曲线')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('lightgbm_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

# 主函数
if __name__ == '__main__':
    print('===== 使用LightGBM预测客户未来3个月资产是否能提升至100万+ =====')
    
    # 1. 加载和预处理数据
    print('\n1. 正在加载和预处理数据...')
    df_merged = load_and_preprocess_data()
    print(f'成功加载 {len(df_merged)} 条客户数据')
    
    # 2. 准备特征和标签
    print('\n2. 正在准备特征和标签...')
    X, y, features = prepare_features_and_labels(df_merged)
    
    # 检查标签分布
    print(f'标签分布: {y.value_counts()}')
    
    if len(y.value_counts()) < 2:
        print("警告：标签分布不均衡，可能影响模型训练效果")
        # 在这种情况下，我们调整阈值以确保有正负样本
        threshold = np.percentile(X['total_assets'], 80)  # 调整为前20%的客户为高价值客户
        y = (df_merged['total_assets'] >= threshold).astype(int)
        print(f'重新设定标签分布: {y.value_counts()}')
    
    # 3. 训练LightGBM模型
    print('\n3. 正在训练LightGBM模型...')
    model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba = train_lightgbm_model(X, y)
    
    # 4. 输出特征重要性排序（文本打印）
    print('\n4. 特征重要性分析:')
    importance_df = print_feature_importance(model, features)
    
    # 5. 生成特征重要性可视化图片
    print('\n5. 生成特征重要性可视化图片...')
    visualize_feature_importance(model, features)
    
    # 6. 模型评估
    evaluate_model(y_test, y_pred, y_pred_proba)
    
    # 7. 绘制ROC曲线
    print('\n7. 绘制ROC曲线...')
    visualize_roc_curve(y_test, y_pred_proba)
    
    # 8. 对新客户进行预测
    predict_new_customers(model, features, X)
    
    print('\n程序执行完毕！')
    print('特征重要性图片已保存为 lightgbm_feature_importance.png')
    print('ROC曲线已保存为 lightgbm_roc_curve.png')
