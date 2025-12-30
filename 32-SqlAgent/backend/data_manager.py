
"""
数据管理器 - 使用pandas管理CSV数据
"""

import pandas as pd
import os
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

class DataManager:
    """数据管理器类"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, Dict] = {}
        self.db_path = os.path.join(self.data_dir, "sales_data.db")
        self._load_all_data()
        self._ensure_database()

    def _load_all_data(self):
        """加载所有数据"""
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)

        # 加载真实的销售数据
        sales_file = os.path.join(self.data_dir, "电子产品销售数据.csv")
        if os.path.exists(sales_file):
            print(f"加载销售数据: {sales_file}")
            df = pd.read_csv(sales_file, encoding='utf-8')
            # 清理列名
            df.columns = [
                "product_name", "price", "sales_volume", "sale_date",
                "category", "brand", "original_price", "discount_rate",
                "review_count", "positive_rate", "ship_location", "warranty_months"
            ]
            self.data_cache["sales_data"] = df
            self.metadata["sales_data"] = {
                "name": "电子产品销售数据",
                "description": "真实的电子产品销售记录",
                "columns": df.columns.tolist(),
                "rows": len(df),
                "source": "csv"
            }

        # 创建模拟的ERP数据
        self._create_erp_data()

        # 创建销售分析数据
        self._create_sales_analysis_data()

        print(f"数据加载完成！共加载 {len(self.data_cache)} 个数据集")

    def _create_erp_data(self):
        """创建模拟的ERP数据"""
        # 产品数据
        erp_products = [
            {"product_id": 1, "name": "iPhone 15 Pro", "category": "手机", "price": 8999.00, "stock": 500, "supplier": "Apple"},
            {"product_id": 2, "name": "MacBook Pro M3", "category": "笔记本", "price": 16999.00, "stock": 200, "supplier": "Apple"},
            {"product_id": 3, "name": "iPad Air", "category": "平板", "price": 4799.00, "stock": 800, "supplier": "Apple"},
            {"product_id": 4, "name": "AirPods Pro", "category": "耳机", "price": 1999.00, "stock": 2000, "supplier": "Apple"},
            {"product_id": 5, "name": "Apple Watch", "category": "智能手表", "price": 3199.00, "stock": 1000, "supplier": "Apple"},
            {"product_id": 6, "name": "Samsung Galaxy S24", "category": "手机", "price": 7999.00, "stock": 600, "supplier": "Samsung"},
            {"product_id": 7, "name": "Sony WH-1000XM5", "category": "耳机", "price": 2999.00, "stock": 1500, "supplier": "Sony"},
            {"product_id": 8, "name": "Dell XPS 15", "category": "笔记本", "price": 13999.00, "stock": 300, "supplier": "Dell"},
            {"product_id": 9, "name": "Surface Pro 9", "category": "平板", "price": 8999.00, "stock": 400, "supplier": "Microsoft"},
            {"product_id": 10, "name": "ThinkPad X1 Carbon", "category": "笔记本", "price": 12999.00, "stock": 350, "supplier": "Lenovo"}
        ]

        products_df = pd.DataFrame(erp_products)
        self.data_cache["erp_products"] = products_df
        self.metadata["erp_products"] = {
            "name": "ERP产品表",
            "description": "企业资源规划系统中的产品数据",
            "columns": products_df.columns.tolist(),
            "rows": len(products_df),
            "source": "mock"
        }

        # 订单数据
        import random
        from datetime import datetime, timedelta

        customers = ['张三', '李四', '王五', '赵六', '陈七', '刘八', '周九', '吴十']
        statuses = ['已完成', '处理中', '待发货']

        orders = []
        for i in range(1, 201):  # 生成200条订单
            product_id = random.randint(1, 10)
            product = next(p for p in erp_products if p["product_id"] == product_id)
            quantity = random.randint(1, 10)
            order_date = (datetime.now() - timedelta(days=random.randint(0, 180))).strftime('%Y-%m-%d')

            orders.append({
                "order_id": i,
                "product_id": product_id,
                "product_name": product["name"],
                "customer_name": random.choice(customers),
                "quantity": quantity,
                "unit_price": product["price"],
                "total_amount": product["price"] * quantity,
                "order_date": order_date,
                "status": random.choice(statuses),
                "category": product["category"]
            })

        orders_df = pd.DataFrame(orders)
        self.data_cache["erp_orders"] = orders_df
        self.metadata["erp_orders"] = {
            "name": "ERP订单表",
            "description": "企业资源规划系统中的订单数据",
            "columns": orders_df.columns.tolist(),
            "rows": len(orders_df),
            "source": "mock"
        }

        # 客户数据
        customers_data = [
            {"customer_id": 1, "name": "张三", "email": "zhangsan@email.com", "city": "北京", "level": "VIP", "total_orders": 25},
            {"customer_id": 2, "name": "李四", "email": "lisi@email.com", "city": "上海", "level": "普通", "total_orders": 18},
            {"customer_id": 3, "name": "王五", "email": "wangwu@email.com", "city": "广州", "level": "VIP", "total_orders": 32},
            {"customer_id": 4, "name": "赵六", "email": "zhaoliu@email.com", "city": "深圳", "level": "普通", "total_orders": 15},
            {"customer_id": 5, "name": "陈七", "email": "chenqi@email.com", "city": "杭州", "level": "VIP", "total_orders": 28},
        ]

        customers_df = pd.DataFrame(customers_data)
        self.data_cache["erp_customers"] = customers_df
        self.metadata["erp_customers"] = {
            "name": "ERP客户表",
            "description": "企业资源规划系统中的客户数据",
            "columns": customers_df.columns.tolist(),
            "rows": len(customers_df),
            "source": "mock"
        }

    def _create_sales_analysis_data(self):
        """创建销售分析数据"""
        # 月度销售汇总
        if "erp_orders" in self.data_cache:
            orders_df = self.data_cache["erp_orders"].copy()
            orders_df["order_date"] = pd.to_datetime(orders_df["order_date"])
            orders_df["month"] = orders_df["order_date"].dt.strftime('%Y-%m')

            monthly_summary = orders_df.groupby(["month", "category"]).agg({
                "total_amount": "sum",
                "quantity": "sum",
                "order_id": "count"
            }).rename(columns={"order_id": "order_count"}).reset_index()

            self.data_cache["monthly_sales"] = monthly_summary
            self.metadata["monthly_sales"] = {
                "name": "月度销售汇总",
                "description": "按月和类别汇总的销售数据",
                "columns": monthly_summary.columns.tolist(),
                "rows": len(monthly_summary),
                "source": "derived"
            }

    def _ensure_database(self):
        """确保SQLite数据库存在并同步数据"""
        try:
            import sqlite3
            
            # 如果数据库不存在，创建它
            if not os.path.exists(self.db_path):
                print(f"Creating database at {self.db_path}...")
                conn = sqlite3.connect(self.db_path)
                
                # 将所有DataFrame写入数据库
                for table_name, df in self.data_cache.items():
                    df.to_sql(table_name, conn, if_exists='replace', index=False)
                    print(f"  - Exported {table_name} ({len(df)} rows)")
                
                conn.close()
                print(f"✅ Database created at {self.db_path}")
            else:
                print(f"✅ Database already exists at {self.db_path}")
                
        except Exception as e:
            print(f"Warning: Could not create database: {e}")

    def get_table_list(self) -> List[Dict[str, Any]]:
        """获取所有数据表列表"""
        table_list = []
        for table_name, metadata in self.metadata.items():
            table_list.append({
                "name": metadata["name"],
                "table": table_name,
                "rows": metadata["rows"],
                "columns": metadata["columns"],
                "description": metadata["description"],
                "source": metadata["source"]
            })
        return table_list

    def get_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """获取表详细信息"""
        if table_name in self.metadata:
            return self.metadata[table_name]
        return None

    def query_data(self, query: str, table_name: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
        """
        执行查询（使用自然语言）
        """
        results = []
        answer = ""

        # 如果指定了表名，只在该表中查询
        if table_name and table_name in self.data_cache:
            df = self.data_cache[table_name]
            results, answer = self._process_query(query, df, table_name)
        else:
            # 在所有表中搜索
            for name, df in self.data_cache.items():
                result, ans = self._process_query(query, df, name)
                if result:
                    results.extend(result)
                    if ans:
                        answer += f"\n[{name}] {ans}"

        # 限制结果数量
        if results and len(results) > limit:
            results = results[:limit]
            answer += f"\n（显示前{limit}条结果）"

        return {
            "success": True,
            "data": results,
            "answer": answer or f"查询完成，返回{len(results)}条结果",
            "columns": list(results[0].keys()) if results else [],
            "total_rows": len(results)
        }

    def _process_query(self, query: str, df: pd.DataFrame, table_name: str) -> tuple:
        """处理查询逻辑"""
        query_lower = query.lower()
        results = []
        answer = ""

        try:
            # 简单的查询逻辑
            if "前" in query and "条" in query or "top" in query_lower:
                # 获取前N条
                n = 10  # 默认10条
                if "5" in query:
                    n = 5
                elif "20" in query:
                    n = 20

                results = df.head(n).to_dict('records')
                answer = f"返回前{n}条数据"

            elif "销售" in query and "额" in query and ("总计" in query or "总和" in query or "总" in query):
                # 计算总销售额
                if "total_amount" in df.columns:
                    total = df["total_amount"].sum()
                    answer = f"总销售额为: {total:,.2f}元"
                elif "price" in df.columns and "sales_volume" in df.columns:
                    total = (df["price"] * df["sales_volume"]).sum()
                    answer = f"总销售额为: {total:,.2f}元"
                results = df.head(5).to_dict('records')

            elif "数量" in query and ("统计" in query or "计数" in query or "多少" in query):
                # 统计数量
                count = len(df)
                answer = f"共有{count}条记录"
                results = df.head(5).to_dict('records')

            elif "分类" in query or "类别" in query:
                # 按类别统计
                if "category" in df.columns:
                    category_count = df["category"].value_counts().to_dict()
                    answer = f"各类别数量: {category_count}"
                    results = [{"category": k, "count": v} for k, v in category_count.items()]
                elif "分类" in df.columns:
                    category_count = df["分类"].value_counts().to_dict()
                    answer = f"各类别数量: {category_count}"
                    results = [{"分类": k, "count": v} for k, v in category_count.items()]

            elif "品牌" in query:
                # 按品牌统计
                if "brand" in df.columns:
                    brand_count = df["brand"].value_counts().to_dict()
                    answer = f"各品牌数量: {brand_count}"
                    results = [{"brand": k, "count": v} for k, v in brand_count.items()]

            elif "最大" in query or "最高" in query:
                # 查找最大值
                if "price" in df.columns:
                    max_idx = df["price"].idxmax()
                    max_row = df.loc[max_idx]
                    results = [max_row.to_dict()]
                    answer = f"价格最高的产品: {max_row.get('product_name', max_row.get('name', '未知'))}"

            elif "最小" in query or "最低" in query:
                # 查找最小值
                if "price" in df.columns:
                    min_idx = df["price"].idxmin()
                    min_row = df.loc[min_idx]
                    results = [min_row.to_dict()]
                    answer = f"价格最低的产品: {min_row.get('product_name', min_row.get('name', '未知'))}"

            else:
                # 默认返回前5条
                results = df.head(5).to_dict('records')
                answer = f"查询'{query}'，返回相关数据"

        except Exception as e:
            answer = f"查询出错: {str(e)}"
            results = df.head(5).to_dict('records')

        return results, answer

# 创建全局数据管理器实例
data_manager = DataManager()

# 测试函数
def test_data_manager():
    """测试数据管理器"""
    print("=== 测试数据管理器 ===")

    # 获取表列表
    tables = data_manager.get_table_list()
    print(f"数据表列表: {len(tables)}个")
    for table in tables:
        print(f"  - {table['name']} ({table['rows']}行)")

    # 测试查询
    print("\n=== 测试查询 ===")

    queries = [
        "显示前5条数据",
        "总销售额是多少",
        "各类别产品数量",
        "价格最高的产品",
        "手机类别的产品"
    ]

    for query in queries:
        print(f"\n查询: {query}")
        result = data_manager.query_data(query)
        print(f"结果: {result['answer']}")
        print(f"返回: {len(result['data'])}条")

if __name__ == "__main__":
    test_data_manager()