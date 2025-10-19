import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import warnings
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 忽略警告信息
warnings.filterwarnings('ignore')

# 读取数据
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

# 构造特征和标签
def prepare_features_and_labels(df):
    """准备模型所需的特征和标签"""
    # 选择业务相关的特征
    features = [
        'total_assets',         # 总资产
        'monthly_income',       # 月收入
        'product_count',        # 持有产品数量
        'app_login_count',      # 手机银行登录次数
        'financial_repurchase_count',  # 理财复购次数
        'investment_monthly_count'     # 月均投资次数
    ]
    
    # 确保所有特征都存在
    existing_features = [f for f in features if f in df.columns]
    print(f"使用的特征: {existing_features}")
    
    # 构造标签：模拟未来3个月资产是否能提升至100万+（在实际业务中，这里应该用真实的未来资产数据）
    np.random.seed(42)  # 设置随机种子，保证结果可复现
    
    # 根据客户当前特征生成未来资产的模拟数据
    # 这里使用了当前资产、月收入和行为特征来影响未来资产增长
    growth_factor = 1.0 + 0.05 * np.random.randn(len(df))  # 基础增长因子
    growth_factor = np.clip(growth_factor, 0.9, 1.3)  # 限制增长范围
    
    # 高活跃客户有更高的增长潜力
    growth_factor = np.where(df['app_login_count'] > df['app_login_count'].median(), 
                            growth_factor * 1.1, growth_factor)
    
    # 高收入客户有更高的增长潜力
    growth_factor = np.where(df['monthly_income'] > df['monthly_income'].median(), 
                            growth_factor * 1.1, growth_factor)
    
    # 计算未来3个月的模拟资产
    df['future_total_assets'] = df['total_assets'] * growth_factor
    
    # 构建标签：未来3个月资产是否达到100万+（1表示是，0表示否）
    df['label'] = (df['future_total_assets'] >= 1000000).astype(int)
    
    # 特征值可能有缺失，用0填充
    X = df[existing_features].fillna(0)
    y = df['label']
    
    return X, y, existing_features

# 训练逻辑回归模型
def train_logistic_regression(X, y):
    """训练逻辑回归模型"""
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练逻辑回归模型
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_train, y_train)
    
    # 模型预测
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return model, X_train, X_test, y_train, y_test, y_pred, y_prob

# 可视化逻辑回归系数
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
    plt.figure(figsize=(10, 6))
    
    # 使用颜色区分正负系数
    colors = ['red' if x < 0 else 'blue' for x in feature_coef['系数']]
    
    # 绘制水平条形图
    bars = plt.barh(feature_coef['特征'], feature_coef['系数'], color=colors)
    
    # 添加垂直参考线
    plt.axvline(0, color='gray', linestyle='--', alpha=0.7)
    
    # 添加标题和标签
    plt.title('逻辑回归特征系数分析（预测未来3个月资产提升至100万+）', fontsize=14)
    plt.xlabel('系数值', fontsize=12)
    plt.ylabel('特征名称', fontsize=12)
    
    # 添加数值标签
    for bar in bars:
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
    plt.savefig('logistic_regression_coefficients.png', dpi=300)
    
    # 显示图表
    plt.show()
    
    return feature_coef

# 评估模型性能
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

# 主函数
if __name__ == '__main__':
    print('===== 逻辑回归预测高价值客户程序 =====')
    
    # 1. 加载和预处理数据
    print('\n1. 正在加载和预处理数据...')
    df_merged = load_and_preprocess_data()
    print(f'成功加载 {len(df_merged)} 条客户数据')
    
    # 2. 准备特征和标签
    print('\n2. 正在准备特征和标签...')
    X, y, features = prepare_features_and_labels(df_merged)
    print(f'标签分布: {y.value_counts()}')
    
    # 3. 训练逻辑回归模型
    print('\n3. 正在训练逻辑回归模型...')
    model, X_train, X_test, y_train, y_test, y_pred, y_prob = train_logistic_regression(X, y)
    
    # 4. 可视化并输出系数
    print('\n4. 逻辑回归系数分析:')
    feature_coef = visualize_coefficients(model, features)
    print('特征系数详情:')
    print(feature_coef.sort_values('系数', ascending=False).to_string(index=False))
    
    # 5. 模型评估
    evaluate_model(y_test, y_pred, y_prob)
    
    # 6. 输出前10个客户的预测概率
    print('\n5. 前10个客户的预测结果:')
    sample_customers = df_merged.head(10).copy()
    sample_customers['预测概率'] = model.predict_proba(X[:10])[:, 1]
    sample_customers['预测类别'] = model.predict(X[:10])
    
    # 显示客户ID、当前资产、预测概率和预测类别
    result_cols = ['customer_id', 'total_assets', '预测概率', '预测类别']
    print(sample_customers[result_cols].to_string(index=False))
    
    print('\n程序执行完毕！系数可视化图表已保存为 logistic_regression_coefficients.png')