import os
from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np


app = Flask(__name__)


# 数据加载
def load_data():
    """加载客户基础数据和行为资产数据"""
    base_df = pd.read_csv('28-AI智慧运营/02百万客群经营(大屏)/customer_base.csv')
    behavior_df = pd.read_csv('28-AI智慧运营/02百万客群经营(大屏)/customer_behavior_assets.csv')
    # 合并数据
    merged_df = pd.merge(base_df, behavior_df, on='customer_id', how='inner')
    return base_df, behavior_df, merged_df


# 加载数据
base_data, behavior_data, merged_data = load_data()


def get_asset_level_distribution():
    """获取资产等级分布数据"""
    asset_dist = merged_data.groupby('asset_level').size().reset_index(name='count')
    return [
        {"name": row['asset_level'], "value": row['count']} 
        for _, row in asset_dist.iterrows()
    ]


def get_lifecycle_product_heatmap():
    """获取生命周期与产品持有热力图数据"""
    # 透视表：生命周期阶段 vs 产品持有
    pivot_data = merged_data.groupby(['lifecycle_stage', 'product_count']).size().reset_index(name='count')
    
    # 获取所有生命周期阶段和产品数量的唯一值
    lifecycle_stages = merged_data['lifecycle_stage'].unique()
    product_counts = merged_data['product_count'].unique()
    
    # 创建坐标数据，确保将numpy类型转换为Python原生类型
    data = []
    for i, stage in enumerate(lifecycle_stages):
        for j, count in enumerate(product_counts):
            value = merged_data[ 
                (merged_data['lifecycle_stage'] == stage) & 
                (merged_data['product_count'] == count) 
            ].shape[0]
            # 转换为Python原生int类型
            data.append([int(j), int(i), int(value)])
    
    return {
        "data": data,
        "xAxis": [int(x) for x in product_counts],  # 转换为Python原生int类型
        "yAxis": list(lifecycle_stages)  # 转换为Python列表
    }


def get_high_potential_customers():
    """获取高潜力客户画像分析"""
    # 定义高潜力客户：总资产高且行为活跃
    high_potential = merged_data[
        (merged_data['total_assets'] > merged_data['total_assets'].quantile(0.8)) &
        (merged_data['app_login_count'] > merged_data['app_login_count'].quantile(0.7))
    ]
    
    # 按年龄分组的统计
    age_grouped = high_potential.groupby('age').agg({
        'customer_id': 'count',
        'total_assets': 'mean'
    }).reset_index()
    
    return {
        "ages": age_grouped['age'].tolist(),
        "customer_counts": age_grouped['customer_id'].tolist(),
        "avg_assets": age_grouped['total_assets'].tolist()
    }


def get_marketing_performance():
    """获取营销触达效果"""
    contact_stats = merged_data['contact_result'].value_counts().reset_index()
    contact_stats.columns = ['result', 'count']
    
    total_contacts = contact_stats['count'].sum()
    success_rate = contact_stats[contact_stats['result'] == '成功']['count'].sum() / total_contacts if total_contacts > 0 else 0
    
    return {
        "results": contact_stats['result'].tolist(),
        "counts": contact_stats['count'].tolist(),
        "success_rate": success_rate
    }


def get_geographic_distribution():
    """获取地域分布与经营情况"""
    geo_stats = merged_data.groupby('city_level').agg({
        'customer_id': 'count',
        'total_assets': 'mean'
    }).reset_index()
    
    return {
        "city_levels": geo_stats['city_level'].tolist(),
        "customer_counts": geo_stats['customer_id'].tolist(),
        "avg_assets": geo_stats['total_assets'].tolist()
    }


def get_dashboard_summary():
    """获取仪表板摘要统计信息"""
    # 基于customer_base.csv计算总客户数（去重）
    total_customers = base_data['customer_id'].nunique()
    
    # 计算平均资产（基于合并数据）
    avg_assets = float(merged_data['total_assets'].mean())
    
    # 计算活跃客户（app_login_count > 5 定义为活跃客户），基于合并数据去重
    active_customers_data = merged_data[merged_data['app_login_count'] > 5]
    active_customers = active_customers_data['customer_id'].nunique()
    active_rate = (active_customers / total_customers) * 100 if total_customers > 0 else 0
    
    return {
        "total_customers": total_customers,
        "avg_assets": avg_assets,
        "active_rate": active_rate
    }


@app.route('/')
def index():
    """主页面"""
    return render_template('dashboard.html')


@app.route('/api/asset_distribution')
def asset_distribution():
    """资产等级分布API"""
    return jsonify(get_asset_level_distribution())


@app.route('/api/lifecycle_product_heatmap')
def lifecycle_product_heatmap():
    """生命周期与产品持有热力图API"""
    return jsonify(get_lifecycle_product_heatmap())


@app.route('/api/high_potential_customers')
def high_potential_customers():
    """高潜力客户画像API"""
    return jsonify(get_high_potential_customers())


@app.route('/api/marketing_performance')
def marketing_performance():
    """营销效果API"""
    return jsonify(get_marketing_performance())


@app.route('/api/geographic_distribution')
def geographic_distribution():
    """地域分布API"""
    return jsonify(get_geographic_distribution())


@app.route('/api/dashboard_summary')
def dashboard_summary():
    """仪表板摘要统计API"""
    return jsonify(get_dashboard_summary())


if __name__ == '__main__':
    app.run(debug=True,port=8080)