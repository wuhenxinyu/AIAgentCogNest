
"""
初始化SQLite数据库，导入CSV数据
"""

import sqlite3
import pandas as pd
import os
from datetime import datetime

# 数据库路径
DB_PATH = "data/sales_data.db"
CSV_PATH = "data/电子产品销售数据.csv"

def create_database():
    """创建数据库并导入数据"""
    print("开始初始化数据库...")

    # 确保data目录存在
    os.makedirs("data", exist_ok=True)

    # 读取CSV数据
    print(f"读取CSV文件: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, encoding='utf-8')

    # 清理列名（确保是有效的SQL标识符）
    df.columns = [
        "product_name", "price", "sales_volume", "sale_date",
        "category", "brand", "original_price", "discount_rate",
        "review_count", "positive_rate", "ship_location", "warranty_months"
    ]

    # 转换日期格式
    df['sale_date'] = pd.to_datetime(df['sale_date'])

    print(f"数据行数: {len(df)}")
    print(f"列名: {df.columns.tolist()}")

    # 创建数据库连接
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 创建销售数据表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sales_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name TEXT NOT NULL,
            price REAL NOT NULL,
            sales_volume INTEGER NOT NULL,
            sale_date TEXT NOT NULL,
            category TEXT NOT NULL,
            brand TEXT NOT NULL,
            original_price REAL NOT NULL,
            discount_rate REAL NOT NULL,
            review_count INTEGER NOT NULL,
            positive_rate REAL NOT NULL,
            ship_location TEXT NOT NULL,
            warranty_months INTEGER NOT NULL
        )
    ''')

    # 插入数据
    print("正在插入数据...")
    df.to_sql('sales_data', conn, if_exists='replace', index=False)

    # 创建ERP产品表（模拟数据）
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS erp_products (
            product_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price DECIMAL(10,2) NOT NULL,
            stock INTEGER NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')

    # 插入模拟的ERP产品数据
    erp_products = [
        (1, 'iPhone 15 Pro', '电子产品', 8999.00, 500, '在售', datetime.now().isoformat()),
        (2, 'MacBook Pro M3', '电子产品', 16999.00, 200, '在售', datetime.now().isoformat()),
        (3, 'iPad Air', '电子产品', 4799.00, 800, '在售', datetime.now().isoformat()),
        (4, 'AirPods Pro', '配件', 1999.00, 2000, '在售', datetime.now().isoformat()),
        (5, 'Apple Watch', '电子产品', 3199.00, 1000, '在售', datetime.now().isoformat()),
        (6, 'Samsung Galaxy S24', '电子产品', 7999.00, 600, '在售', datetime.now().isoformat()),
        (7, 'Sony WH-1000XM5', '配件', 2999.00, 1500, '在售', datetime.now().isoformat()),
        (8, 'Dell XPS 15', '电子产品', 13999.00, 300, '在售', datetime.now().isoformat()),
        (9, 'Surface Pro 9', '电子产品', 8999.00, 400, '在售', datetime.now().isoformat()),
        (10, 'ThinkPad X1 Carbon', '电子产品', 12999.00, 350, '在售', datetime.now().isoformat())
    ]

    cursor.executemany('''
        INSERT OR REPLACE INTO erp_products
        (product_id, name, category, price, stock, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', erp_products)

    # 创建订单表（模拟数据）
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS erp_orders (
            order_id INTEGER PRIMARY KEY,
            product_id INTEGER NOT NULL,
            customer_name TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            total_amount DECIMAL(10,2) NOT NULL,
            order_date TEXT NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY (product_id) REFERENCES erp_products (product_id)
        )
    ''')

    # 生成模拟订单数据
    import random
    from datetime import datetime, timedelta

    customers = ['张三', '李四', '王五', '赵六', '陈七', '刘八', '周九', '吴十']
    statuses = ['已完成', '处理中', '待发货']

    orders = []
    for i in range(1, 101):  # 生成100条订单
        product_id = random.randint(1, 10)
        product = next(p for p in erp_products if p[0] == product_id)
        quantity = random.randint(1, 10)
        order_date = (datetime.now() - timedelta(days=random.randint(0, 90))).isoformat()

        orders.append((
            i,
            product_id,
            random.choice(customers),
            quantity,
            product[3] * quantity,  # price * quantity
            order_date,
            random.choice(statuses)
        ))

    cursor.executemany('''
        INSERT OR REPLACE INTO erp_orders
        (order_id, product_id, customer_name, quantity, total_amount, order_date, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', orders)

    # 创建客户表（模拟数据）
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS erp_customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT,
            city TEXT NOT NULL,
            registration_date TEXT NOT NULL,
            total_orders INTEGER DEFAULT 0,
            total_spent DECIMAL(10,2) DEFAULT 0
        )
    ''')

    customers_data = [
        (1, '张三', 'zhangsan@email.com', '13800138001', '北京', '2024-01-15', 25, 125000.00),
        (2, '李四', 'lisi@email.com', '13800138002', '上海', '2024-02-20', 18, 89000.00),
        (3, '王五', 'wangwu@email.com', '13800138003', '广州', '2024-03-10', 32, 156000.00),
        (4, '赵六', 'zhaoliu@email.com', '13800138004', '深圳', '2024-04-05', 15, 67000.00),
        (5, '陈七', 'chenqi@email.com', '13800138005', '杭州', '2024-05-12', 28, 134000.00),
    ]

    cursor.executemany('''
        INSERT OR REPLACE INTO erp_customers
        (customer_id, name, email, phone, city, registration_date, total_orders, total_spent)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', customers_data)

    # 提交事务
    conn.commit()
    conn.close()

    print(f"✅ 数据库初始化完成！")
    print(f"   数据库路径: {DB_PATH}")
    print(f"   销售数据: {len(df)} 条")
    print(f"   ERP产品: {len(erp_products)} 条")
    print(f"   ERP订单: {len(orders)} 条")
    print(f"   ERP客户: {len(customers_data)} 条")

    # 创建索引以提高查询性能
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_sales_category ON sales_data(category)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_sales_brand ON sales_data(brand)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_sales_date ON sales_data(sale_date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_orders_product ON erp_orders(product_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_orders_date ON erp_orders(order_date)')

    conn.commit()
    conn.close()

    print("✅ 索引创建完成")

def test_database():
    """测试数据库连接和查询"""
    print("\n测试数据库连接...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 获取所有表
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"数据库中的表: {[t[0] for t in tables]}")

    # 测试查询
    print("\n查询示例数据:")
    cursor.execute("SELECT COUNT(*) FROM sales_data")
    count = cursor.fetchone()[0]
    print(f"销售数据总记录数: {count}")

    cursor.execute("SELECT category, COUNT(*) FROM sales_data GROUP BY category")
    categories = cursor.fetchall()
    print(f"各类别数量: {categories}")

    cursor.execute("SELECT name, price, stock FROM erp_products LIMIT 5")
    products = cursor.fetchall()
    print(f"前5个产品: {products[:3]}")

    conn.close()
    print("✅ 数据库测试成功")

if __name__ == "__main__":
    try:
        create_database()
        test_database()
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()