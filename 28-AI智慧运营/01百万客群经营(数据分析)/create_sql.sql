-- 客户基础信息表
CREATE TABLE IF NOT EXISTS customer_base (
    customer_id VARCHAR(32) PRIMARY KEY COMMENT '客户ID',
    name VARCHAR(100) COMMENT '客户姓名',
    age INT COMMENT '年龄',
    gender VARCHAR(10) COMMENT '性别',
    occupation VARCHAR(100) COMMENT '职业',
    occupation_type VARCHAR(50) COMMENT '职业类型标签（如：企业高管/互联网从业者/私营业主）',
    monthly_income DECIMAL(12,2) COMMENT '月收入',
    open_account_date VARCHAR(10) COMMENT '开户日期',
    lifecycle_stage VARCHAR(50) COMMENT '客户生命周期',
    marriage_status VARCHAR(20) COMMENT '婚姻状态',
    city_level VARCHAR(20) COMMENT '城市等级（一线/二线城市）',
    branch_name VARCHAR(100) COMMENT '开户网点'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='客户基础信息表';

-- 创建索引
CREATE INDEX idx_occupation_type ON customer_base(occupation_type);
CREATE INDEX idx_lifecycle_stage ON customer_base(lifecycle_stage);
CREATE INDEX idx_city_level ON customer_base(city_level);

-- 客户行为资产表
CREATE TABLE IF NOT EXISTS customer_behavior_assets (
    id VARCHAR(32) PRIMARY KEY COMMENT '主键ID',
    customer_id VARCHAR(32) COMMENT '客户ID（关联customer_base表）',
    
    -- 资产相关
    total_assets DECIMAL(16,2) COMMENT '总资产',
    deposit_balance DECIMAL(16,2) COMMENT '存款余额',
    financial_balance DECIMAL(16,2) COMMENT '理财余额',
    fund_balance DECIMAL(16,2) COMMENT '基金余额',
    insurance_balance DECIMAL(16,2) COMMENT '保险余额',
    asset_level VARCHAR(20) COMMENT '资产分层（50万以下、50-80万、80-100万、100万+）',
    
    -- 产品持有
    deposit_flag TINYINT COMMENT '是否持有存款（1是0否）',
    financial_flag TINYINT COMMENT '是否持有理财（1是0否）',
    fund_flag TINYINT COMMENT '是否持有基金（1是0否）',
    insurance_flag TINYINT COMMENT '是否持有保险（1是0否）',
    product_count INT COMMENT '持有产品数量',
    
    -- 交易行为
    financial_repurchase_count INT COMMENT '近1年理财复购次数',
    credit_card_monthly_expense DECIMAL(12,2) COMMENT '信用卡月均消费',
    investment_monthly_count INT COMMENT '月均投资交易次数',
    
    -- APP行为
    app_login_count INT COMMENT 'APP月均登录次数',
    app_financial_view_time INT COMMENT '理财页面月均停留时长(秒)',
    app_product_compare_count INT COMMENT '产品对比点击次数',
    last_app_login_time VARCHAR(19) NULL COMMENT '最近APP登录时间',
    
    -- 营销触达
    last_contact_time VARCHAR(19) NULL COMMENT '最近联系时间',
    contact_result VARCHAR(50) COMMENT '联系结果',
    marketing_cool_period VARCHAR(10) COMMENT '营销冷却期（30天）',
    
    stat_month VARCHAR(7) COMMENT '统计月份（YYYY-MM）',
    
    -- 外键约束
    FOREIGN KEY (customer_id) REFERENCES customer_base(customer_id),
    
    -- 创建联合唯一索引，确保每个客户每月只有一条记录
    UNIQUE KEY uk_customer_month (customer_id, stat_month)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='客户行为资产表';

-- 创建索引
CREATE INDEX idx_asset_level ON customer_behavior_assets(asset_level);
CREATE INDEX idx_stat_month ON customer_behavior_assets(stat_month);
CREATE INDEX idx_marketing_cool_period ON customer_behavior_assets(marketing_cool_period); 