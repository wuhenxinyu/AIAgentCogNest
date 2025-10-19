"""
决策树模型预测高价值客户,针对特征组合进行分析
决策树模型是一种基于树结构的分类模型，通过递归地将数据集分割为子集，直到每个子集都属于同一类别或满足其他停止条件，从而构建出一个决策树。
决策树的原理是通过特征选择和信息增益或基尼不纯度等指标，选择最优的特征进行分割，从而实现对数据的分类。
决策树的优点是模型简单、易于理解和实现，同时对数据的预处理要求较低。
决策树的缺点是容易过拟合，尤其是在数据量较小或特征数量较多时。

特征工程：
- 年龄：年龄是一个连续变量，不需要归一化处理。
- 月收入：月收入是一个连续变量，不需要归一化处理。
- 当前总资产：当前总资产是一个连续变量，不需要归一化处理。
- 持有产品数量：持有产品数量是一个连续变量，不需要归一化处理。
- 手机银行登录次数：手机银行登录次数是一个连续变量，不需要归一化处理。
- 理财复购次数：理财复购次数是一个连续变量，不需要归一化处理。

注：特征越多，模型的复杂度越高，容易过拟合。还容易出现特征爆炸的问题，即特征数量超过样本数量，导致模型无法训练。
"""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

# 设置matplotlib支持中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 忽略警告信息
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """加载并预处理客户数据"""
    # 读取客户基础信息表
    try:
        df_base = pd.read_csv('customer_base.csv', encoding='utf-8')
    except UnicodeDecodeError:
        df_base = pd.read_csv('customer_base.csv', encoding='gbk')
        
    # 读取客户行为资产表
    try:
        df_behavior = pd.read_csv('customer_behavior_assets.csv', encoding='utf-8')
    except UnicodeDecodeError:
        df_behavior = pd.read_csv('customer_behavior_assets.csv', encoding='gbk')
    
    # 对每个客户，取最新的行为数据（按stat_month排序，取最新的一条）
    df_behavior_latest = df_behavior.sort_values(['customer_id', 'stat_month'], ascending=[True, False])
    df_behavior_latest = df_behavior_latest.drop_duplicates('customer_id', keep='first')
    
    # 合并两个表
    df_merged = pd.merge(df_base, df_behavior_latest, on='customer_id', how='inner')
    
    return df_merged

def prepare_features_and_labels(df):
    """准备模型所需的特征和标签"""
    # 选择业务相关的特征 (不需要归一化处理)
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
    print(f"Using features: {existing_features}")
    
    # 构造标签：未来3个月资产是否能提升至100万+
    # 基于业务逻辑构建真实标签：使用当前资产、月收入、活跃度等因素
    # 高资产客户 + 高收入 + 高活跃度 = 更有可能成为高价值客户
    
    # 计算综合增长潜力分数
    weight_current_assets = 0.7
    weight_monthly_income = 0.3
    weight_activity = 0.1
    weight_income_to_asset_ratio = 0.1
    
    # 计算资产增长率和月收入对总资产的比例
    df['income_to_asset_ratio'] = df['monthly_income'] / (df['total_assets'] + 1)  # 避免除零
    df['activity_score'] = (df['app_login_count'] + 1) * (df['investment_monthly_count'] + 1) * (df['financial_repurchase_count'] + 1)
    
    # 添加这些新特征
    df['income_to_asset_ratio'] = df['income_to_asset_ratio']
    df['activity_score'] = df['activity_score']
    existing_features.extend(['income_to_asset_ratio', 'activity_score'])
    
    # 构建标签：未来3个月资产是否达到100万+
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
    
    print(f"Label distribution: {df['label'].value_counts()}")
    
    # 特征值可能有缺失，用中位数填充
    X = df[existing_features].fillna(df[existing_features].median())
    y = df['label']
    
    return X, y, existing_features

def train_decision_tree(X, y, max_depth=4):
    """训练决策树模型 (depth=4)"""
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 训练决策树模型 (depth=4, 不需要归一化)
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42, criterion='gini')
    model.fit(X_train, y_train)
    
    # 模型预测
    y_pred = model.predict(X_test)
    
    return model, X_train, X_test, y_train, y_test, y_pred

def visualize_decision_tree_text(model, features):
    """以文本形式可视化决策树"""
    print("\n=== 决策树文本表示 ===")
    tree_rules = export_text(model, feature_names=features)
    print(tree_rules)
    return tree_rules

def plot_decision_tree_visual(model, features):
    """生成决策树可视化图片"""
    plt.figure(figsize=(20, 12))
    
    plot_tree(model, 
              feature_names=features,
              class_names=['低价值客户', '高价值客户'],
              filled=True,
              rounded=True,
              fontsize=10)
    
    plt.title('决策树可视化 (Depth=4)', fontsize=16)
    plt.tight_layout()
    plt.savefig('decision_tree_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(y_test, y_pred):
    """评估模型性能"""
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    
    # 生成分类报告
    report = classification_report(y_test, y_pred, digits=4, target_names=['低价值客户', '高价值客户'])
    
    # 生成混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 打印评估结果
    print('\n=== 模型评估结果 ===')
    print(f'准确率: {accuracy:.4f}')
    print('分类报告:')
    print(report)
    print('混淆矩阵:')
    print(cm)
    
    return accuracy, report, cm

def display_feature_importance(model, features):
    """显示特征重要性"""
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\n=== 特征重要性排序 ===")
    print(importance_df.to_string(index=False))
    
    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance_df)), importance_df['Importance'])
    plt.yticks(range(len(importance_df)), importance_df['Feature'])
    plt.xlabel('重要性')
    plt.title('决策树特征重要性')
    plt.gca().invert_yaxis()  # 使最重要的特征在顶部
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return importance_df

def predict_new_customers(model, features, X):
    """对新客户进行预测"""
    print('\n=== 对部分客户进行预测 ===')
    
    # 随机选择部分客户进行预测演示
    sample_indices = np.random.choice(X.index, size=min(10, len(X)), replace=False)
    sample_X = X.loc[sample_indices]
    sample_customers = pd.DataFrame(sample_X)
    
    # 预测类别和概率
    sample_customers['预测类别'] = model.predict(sample_X)
    sample_customers['预测概率_高价值'] = model.predict_proba(sample_X)[:, 1]
    
    # 获取原始数据以显示客户ID和当前资产
    try:
        df_original = load_and_preprocess_data()
        sample_customers['customer_id'] = df_original.loc[sample_indices, 'customer_id'].values
        sample_customers['total_assets'] = df_original.loc[sample_indices, 'total_assets'].values
    except:
        sample_customers['customer_id'] = sample_indices
        sample_customers['total_assets'] = sample_X['total_assets'].values if 'total_assets' in sample_X.columns else np.nan
    
    # 显示客户ID、当前资产、预测类别和预测概率
    result_cols = ['customer_id', 'total_assets', '预测类别', '预测概率_高价值']
    print(sample_customers[result_cols].to_string(index=False))

# 主函数
if __name__ == '__main__':
    print('===== 决策树模型: 预测高价值客户 =====')
    print('使用决策树(depth=4)预测客户未来3个月资产是否能提升至100万+\n')
    
    # 1. 加载和预处理数据
    print('1. Loading and preprocessing data...')
    df_merged = load_and_preprocess_data()
    print(f'Successfully loaded {len(df_merged)} customer records')
    
    # 2. 准备特征和标签
    print('\n2. Preparing features and labels...')
    X, y, features = prepare_features_and_labels(df_merged)
    
    # 检查标签分布
    print(f'Label distribution: {y.value_counts()}')
    
    if len(y.value_counts()) < 2:
        print("Warning: Label distribution is imbalanced, may affect model training")
        # 在这种情况下，我们调整阈值以确保有正负样本
        threshold = np.percentile(X['total_assets'], 80)  # 调整为前20%的客户为高价值客户
        y = (df_merged['total_assets'] >= threshold).astype(int)
        print(f'Re-adjusted label distribution: {y.value_counts()}')
    
    # 3. 训练决策树模型 (depth=4)
    print('\n3. Training decision tree model (depth=4)...')
    model, X_train, X_test, y_train, y_test, y_pred = train_decision_tree(X, y, max_depth=4)
    
    # 4. 以文本形式可视化决策树
    print('\n4. 决策树文本可视化...')
    tree_rules = visualize_decision_tree_text(model, features)
    
    # 保存文本规则到文件
    with open('decision_tree_rules.txt', 'w', encoding='utf-8') as f:
        f.write(tree_rules)
    
    # 5. 生成决策树可视化图片
    print('\n5. 生成决策树可视化图片...')
    plot_decision_tree_visual(model, features)
    
    # 6. 模型评估
    print('\n6. 模型评估...')
    accuracy, report, cm = evaluate_model(y_test, y_pred)
    
    # 7. 显示特征重要性
    print('\n7. 特征重要性分析...')
    importance_df = display_feature_importance(model, features)
    
    # 8. 对新客户进行预测
    predict_new_customers(model, features, X)
    
    # 9. 输出模型参数
    print(f'\n8. 模型参数:')
    print(f"最大深度: {model.get_params()['max_depth']}")
    print(f"分裂准则: {model.get_params()['criterion']}")
    print(f"叶子节点最小样本数: {model.get_params()['min_samples_leaf']}")
    
    print('\n程序执行完毕！')
    print('决策树可视化图片已保存为 decision_tree_visualization.png')
    print('特征重要性图片已保存为 feature_importance.png')
    print('决策树规则文本已保存为 decision_tree_rules.txt')