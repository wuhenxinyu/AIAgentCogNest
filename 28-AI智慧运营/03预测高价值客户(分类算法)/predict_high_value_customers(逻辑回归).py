"""
逻辑回归（是线性模型）模型预测高价值客户，同时采用时间序列分析。逻辑回归是针对单个特征的 相关性进行分析。

逻辑回归的优点:
1. 模型简单,易于理解和实现
2. 计算效率高,尤其在特征数量很多时
3. 模型解释性强,可以直接解释特征的系数含义
4. 对缺失值不敏感,可以直接处理

AUC（Area Under the Curve）曲线说明:
1. AUC是用于评估分类模型性能的重要指标
2. 它展示了模型在不同阈值下的真阳性率（TPR）和假阳性率（FPR）
3. 曲线下的面积（AUC）越大，模型的分类能力越强
4. 一般来说，AUC值越接近1，模型的分类效果越好

AUC的优点:
1. 不依赖于分类阈值,可以直接评估模型性能
2. 不受样本类别不平衡问题的影响
3. 可以直接比较不同模型的性能

AUC的缺点:
1. 不考虑具体的分类结果,只关注分类能力
2. 对异常值敏感,异常值会严重影响AUC值
3. 无法直接解释AUC值的具体含义,需要结合其他指标（如准确率、F1值等）
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import warnings
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve
import os

# 设置中文字体显示
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
    print(f"使用的特征: {existing_features}")
    
    # 构造标签：未来3个月资产是否能提升至100万+
    # 使用当前资产、月收入、活跃度等信息构建真实业务场景的预测标签
    # 高资产客户 + 高收入 + 高活跃度 = 更有可能成为高价值客户

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
    
    # 为了创建不平衡数据集（更接近真实场景），我们设定阈值
    threshold = np.percentile(potential_score, 95)  # 设定前5%的客户为高价值客户
    
    # 构建标签：未来3个月资产是否达到100万+
    df['label'] = (potential_score >= threshold).astype(int)
    
    print(f"标签分布: {df['label'].value_counts()}")
    
    # 特征值可能有缺失，用中位数填充
    X = df[existing_features].fillna(df[existing_features].median())
    y = df['label']
    
    return X, y, existing_features

def train_logistic_regression(X, y):
    """训练逻辑回归模型"""
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 训练逻辑回归模型
    model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
    model.fit(X_train, y_train)
    
    # 模型预测
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return model, X_train, X_test, y_train, y_test, y_pred, y_prob

def visualize_coefficients(model, features):
    """可视化逻辑回归的系数，展示特征的正负影响"""
    # 获取系数
    coef = model.coef_[0]
    
    # 创建DataFrame存储特征和对应的系数
    feature_coef = pd.DataFrame({
        '特征': features,
        '系数': coef
    })
    
    # 按系数大小排序
    feature_coef = feature_coef.sort_values('系数')
    
    # 创建可视化图表
    plt.figure(figsize=(12, 8))
    
    # 使用颜色区分正负系数
    colors = ['red' if x < 0 else 'blue' for x in feature_coef['系数']]
    
    # 绘制水平条形图
    bars = plt.barh(range(len(feature_coef)), feature_coef['系数'], color=colors)
    
    # 添加垂直参考线
    plt.axvline(0, color='gray', linestyle='--', alpha=0.7)
    
    # 设置y轴标签
    plt.yticks(range(len(feature_coef)), feature_coef['特征'])
    
    # 添加标题和标签
    plt.title('逻辑回归特征系数分析（预测未来3个月资产提升至100万+）', fontsize=14)
    plt.xlabel('系数值', fontsize=12)
    plt.ylabel('特征名称', fontsize=12)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width * 1.05 if width > 0 else width * 1.2, 
                 bar.get_y() + bar.get_height()/2, 
                 f'{width:.4f}', 
                 va='center', 
                 fontsize=10)
    
    # 添加解释说明
    plt.figtext(0.5, 0.01, 
                '注：蓝色表示正影响（特征值越大，预测概率越高），红色表示负影响（特征值越大，预测概率越低）', 
                ha='center', fontsize=9)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图片
    plt.savefig('logistic_regression_coefficients_new.png', dpi=300, bbox_inches='tight')
    
    # 显示图表
    plt.show()
    
    return feature_coef

def visualize_roc_curve(y_test, y_prob):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc_score(y_test, y_prob):.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (False Positive Rate)')
    plt.ylabel('真正率 (True Positive Rate)')
    plt.title('逻辑回归模型ROC曲线')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('logistic_regression_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(y_test, y_pred, y_prob):
    """评估模型性能"""
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    
    # 计算AUC值
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = None
    
    # 生成分类报告
    report = classification_report(y_test, y_pred, digits=4)
    
    # 打印评估结果
    print('\n=== 模型评估结果 ===')
    print(f'准确率: {accuracy:.4f}')
    if auc is not None:
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
    sample_customers['预测概率'] = model.predict_proba(sample_X)[:, 1]
    sample_customers['预测类别'] = model.predict(sample_X)
    
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

# 主函数
if __name__ == '__main__':
    print('===== 预测客户未来3个月资产提升至100万+的逻辑回归模型 =====')
    
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
    
    # 3. 训练逻辑回归模型
    print('\n3. 正在训练逻辑回归模型...')
    model, X_train, X_test, y_train, y_test, y_pred, y_prob = train_logistic_regression(X, y)
    
    # 4. 可视化并输出系数
    print('\n4. 逻辑回归系数分析:')
    feature_coef = visualize_coefficients(model, features)
    print('特征系数详情（按系数大小排序）:')
    print(feature_coef.sort_values('系数', ascending=False).to_string(index=False))
    
    # 5. 模型评估
    evaluate_model(y_test, y_pred, y_prob)
    
    # 6. 绘制ROC曲线
    print('\n6. 绘制ROC曲线...')
    visualize_roc_curve(y_test, y_prob)
    
    # 7. 对新客户进行预测
    predict_new_customers(model, features, X)
    
    # 8. 输出模型的截距
    print(f'\n7. 模型截距: {model.intercept_[0]:.4f}')
    
    print('\n程序执行完毕！系数可视化图表已保存为 logistic_regression_coefficients_new.png')
    print('ROC曲线已保存为 logistic_regression_roc_curve.png')