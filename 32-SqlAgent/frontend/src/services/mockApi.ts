// 模拟数据服务
export interface DataSource {
  id: string;
  name: string;
  type: 'mysql' | 'postgresql' | 'oracle' | 'sqlserver';
  host: string;
  status: 'connected' | 'disconnected';
  tables: TableInfo[];
}

export interface TableInfo {
  name: string;
  rowCount: number;
  columns: ColumnInfo[];
  tableName?: string; // 添加实际的数据库表名
}

export interface ColumnInfo {
  name: string;
  type: string;
  nullable: boolean;
  isPrimaryKey: boolean;
}

export interface QueryResult {
  columns: { name: string; type: string }[];
  rows: Record<string, any>[];
  executionTime: number;
  totalRows: number;
}

// 模拟数据源
const mockDataSources: DataSource[] = [
  {
    id: '1',
    name: 'ERP生产数据库',
    type: 'mysql',
    host: '192.168.1.100',
    status: 'connected',
    tables: [
      {
        name: 'Products',
        rowCount: 1250,
        columns: [
          { name: 'product_id', type: 'INT', nullable: false, isPrimaryKey: true },
          { name: 'name', type: 'VARCHAR', nullable: false, isPrimaryKey: false },
          { name: 'category', type: 'VARCHAR', nullable: true, isPrimaryKey: false },
          { name: 'price', type: 'DECIMAL', nullable: false, isPrimaryKey: false },
          { name: 'stock', type: 'INT', nullable: false, isPrimaryKey: false },
        ]
      },
      {
        name: 'Orders',
        rowCount: 5480,
        columns: [
          { name: 'order_id', type: 'INT', nullable: false, isPrimaryKey: true },
          { name: 'product_id', type: 'INT', nullable: false, isPrimaryKey: false },
          { name: 'customer_id', type: 'INT', nullable: false, isPrimaryKey: false },
          { name: 'quantity', type: 'INT', nullable: false, isPrimaryKey: false },
          { name: 'total_amount', type: 'DECIMAL', nullable: false, isPrimaryKey: false },
          { name: 'order_date', type: 'DATETIME', nullable: false, isPrimaryKey: false },
        ]
      },
      {
        name: 'Customers',
        rowCount: 892,
        columns: [
          { name: 'customer_id', type: 'INT', nullable: false, isPrimaryKey: true },
          { name: 'name', type: 'VARCHAR', nullable: false, isPrimaryKey: false },
          { name: 'email', type: 'VARCHAR', nullable: true, isPrimaryKey: false },
          { name: 'phone', type: 'VARCHAR', nullable: true, isPrimaryKey: false },
          { name: 'city', type: 'VARCHAR', nullable: true, isPrimaryKey: false },
        ]
      }
    ]
  },
  {
    id: '2',
    name: '销售分析库',
    type: 'postgresql',
    host: '192.168.1.101',
    status: 'connected',
    tables: [
      {
        name: 'Sales2023',
        rowCount: 15680,
        columns: [
          { name: 'id', type: 'SERIAL', nullable: false, isPrimaryKey: true },
          { name: 'region', type: 'VARCHAR', nullable: false, isPrimaryKey: false },
          { name: 'product', type: 'VARCHAR', nullable: false, isPrimaryKey: false },
          { name: 'sales_amount', type: 'NUMERIC', nullable: false, isPrimaryKey: false },
          { name: 'month', type: 'VARCHAR', nullable: false, isPrimaryKey: false },
        ]
      },
      {
        name: 'Targets2024',
        rowCount: 48,
        columns: [
          { name: 'target_id', type: 'SERIAL', nullable: false, isPrimaryKey: true },
          { name: 'product_category', type: 'VARCHAR', nullable: false, isPrimaryKey: false },
          { name: 'target_amount', type: 'NUMERIC', nullable: false, isPrimaryKey: false },
          { name: 'quarter', type: 'VARCHAR', nullable: false, isPrimaryKey: false },
        ]
      }
    ]
  },
  {
    id: '3',
    name: '日志数据库',
    type: 'oracle',
    host: '192.168.1.102',
    status: 'disconnected',
    tables: []
  }
];

// 模拟查询结果
const mockQueryResults: Record<string, QueryResult> = {
  'top10': {
    columns: [
      { name: '产品名称', type: 'VARCHAR' },
      { name: '销售数量', type: 'INT' },
      { name: '销售金额', type: 'DECIMAL' }
    ],
    rows: [
      { '产品名称': 'iPhone 15 Pro', '销售数量': 1250, '销售金额': 1498800 },
      { '产品名称': 'MacBook Air M2', '销售数量': 890, '销售金额': 1078000 },
      { '产品名称': 'iPad Pro 12.9', '销售数量': 756, '销售金额': 907200 },
      { '产品名称': 'AirPods Pro 2', '销售数量': 2340, '销售金额': 702000 },
      { '产品名称': 'Apple Watch Series 9', '销售数量': 1120, '销售金额': 560000 },
      { '产品名称': 'Mac mini M2', '销售数量': 450, '销售金额': 399600 },
      { '产品名称': 'Magic Keyboard', '销售数量': 680, '销售金额': 272000 },
      { '产品名称': 'Apple TV 4K', '销售数量': 320, '销售金额': 96000 },
      { '产品名称': 'HomePod mini', '销售数量': 560, '销售金额': 112000 },
      { '产品名称': 'AirTag 4件装', '销售数量': 1890, '销售金额': 56700 }
    ],
    executionTime: 125,
    totalRows: 10
  },
  'monthly': {
    columns: [
      { name: '月份', type: 'VARCHAR' },
      { name: '订单数', type: 'INT' },
      { name: '销售额', type: 'DECIMAL' },
      { name: '平均客单价', type: 'DECIMAL' }
    ],
    rows: [
      { '月份': '2024-01', '订单数': 450, '销售额': 234500, '平均客单价': 521.11 },
      { '月份': '2024-02', '订单数': 380, '销售额': 198300, '平均客单价': 521.84 },
      { '月份': '2024-03', '订单数': 520, '销售额': 287600, '平均客单价': 553.08 },
      { '月份': '2024-04', '订单数': 490, '销售额': 265400, '平均客单价': 541.63 },
      { '月份': '2024-05', '订单数': 580, '销售额': 312500, '平均客单价': 538.79 },
      { '月份': '2024-06', '订单数': 620, '销售额': 356800, '平均客单价': 575.48 },
      { '月份': '2024-07', '订单数': 590, '销售额': 328900, '平均客单价': 557.46 },
      { '月份': '2024-08', '订单数': 640, '销售额': 378200, '平均客单价': 590.94 },
      { '月份': '2024-09', '订单数': 710, '销售额': 425600, '平均客单价': 599.44 },
      { '月份': '2024-10', '订单数': 680, '销售额': 398700, '平均客单价': 586.33 },
      { '月份': '2024-11', '订单数': 750, '销售额': 456800, '平均客单价': 609.07 },
      { '月份': '2024-12', '订单数': 820, '销售额': 512300, '平均客单价': 624.76 }
    ],
    executionTime: 89,
    totalRows: 12
  }
};

// API 模拟函数
export const mockApi = {
  // 获取所有数据源（直接从后端加载）
  async getDataSources(): Promise<DataSource[]> {
    try {
      // 从真实后端获取数据源
      const response = await fetch('/api/datasources');
      const data = await response.json();

      if (data.success && data.sources) {
        // 将后端数据源转换为前端格式，按source分组
        const allDataSources: DataSource[] = [];

        // 按source分组
        const sourceGroups: Record<string, any[]> = {};
        data.sources.forEach((source: any) => {
          const groupKey = source.source;
          if (!sourceGroups[groupKey]) {
            sourceGroups[groupKey] = [];
          }
          sourceGroups[groupKey].push(source);
        });

        // CSV真实数据源组
        if (sourceGroups.csv && sourceGroups.csv.length > 0) {
          allDataSources.push({
            id: 'csv_data',
            name: '真实数据（CSV）',
            type: 'mysql',
            host: 'localhost:8001',
            status: 'connected',
            tables: sourceGroups.csv.map(s => ({
              name: s.name,
              tableName: s.table, // 保存实际的表名
              rowCount: s.rows,
              columns: s.columns.map((col: string) => ({
                name: col,
                type: 'VARCHAR',
                nullable: true,
                isPrimaryKey: false
              }))
            }))
          });
        }

        // Mock ERP数据源组
        if (sourceGroups.mock && sourceGroups.mock.length > 0) {
          allDataSources.push({
            id: 'erp_data',
            name: 'ERP模拟数据',
            type: 'mysql',
            host: 'localhost:8001',
            status: 'connected',
            tables: sourceGroups.mock.map(s => ({
              name: s.name,
              tableName: s.table, // 保存实际的表名
              rowCount: s.rows,
              columns: s.columns.map((col: string) => ({
                name: col,
                type: 'VARCHAR',
                nullable: true,
                isPrimaryKey: false
              }))
            }))
          });
        }

        // Derived汇总数据源组
        if (sourceGroups.derived && sourceGroups.derived.length > 0) {
          allDataSources.push({
            id: 'derived_data',
            name: '汇总数据',
            type: 'mysql',
            host: 'localhost:8001',
            status: 'connected',
            tables: sourceGroups.derived.map(s => ({
              name: s.name,
              tableName: s.table, // 保存实际的表名
              rowCount: s.rows,
              columns: s.columns.map((col: string) => ({
                name: col,
                type: 'VARCHAR',
                nullable: true,
                isPrimaryKey: false
              }))
            }))
          });
        }

        // 直接返回后端数据，不要再添加前端的模拟数据
        return allDataSources;
      }
    } catch (error) {
      console.error('从后端加载数据源失败，使用模拟数据:', error);
    }

    // 如果后端失败，返回模拟数据
    return mockDataSources;
  },

  // 连接数据源
  async connectDataSource(id: string): Promise<boolean> {
    await new Promise(resolve => setTimeout(resolve, 1000));
    const ds = mockDataSources.find(d => d.id === id);
    if (ds) {
      ds.status = 'connected';
      return true;
    }
    return false;
  },

  // 断开数据源
  async disconnectDataSource(id: string): Promise<boolean> {
    await new Promise(resolve => setTimeout(resolve, 500));
    const ds = mockDataSources.find(d => d.id === id);
    if (ds) {
      ds.status = 'disconnected';
      return true;
    }
    return false;
  },

  // 执行查询
  async executeQuery(query: string): Promise<QueryResult> {
    await new Promise(resolve => setTimeout(resolve, 1500));

    // 根据查询内容返回不同的模拟结果
    if (query.includes('Top10') || query.includes('top10')) {
      return mockQueryResults.top10;
    } else if (query.includes('月') || query.includes('month')) {
      return mockQueryResults.monthly;
    } else {
      // 返回默认结果
      return {
        columns: [
          { name: '字段1', type: 'VARCHAR' },
          { name: '字段2', type: 'INT' },
          { name: '字段3', type: 'DECIMAL' }
        ],
        rows: [
          { '字段1': '示例数据1', '字段2': 100, '字段3': 1234.56 },
          { '字段1': '示例数据2', '字段2': 200, '字段3': 2345.67 },
          { '字段1': '示例数据3', '字段2': 300, '字段3': 3456.78 }
        ],
        executionTime: 100,
        totalRows: 3
      };
    }
  },

  // 生成SQL
  async generateSQL(question: string): Promise<{ sql: string; reasoning: string[] }> {
    await new Promise(resolve => setTimeout(resolve, 2000));

    // 简单的规则匹配生成SQL
    if (question.includes('销量') && question.includes('Top10')) {
      return {
        sql: `SELECT
  p.name AS 产品名称,
  SUM(o.quantity) AS 销售数量,
  SUM(o.total_amount) AS 销售金额
FROM ERP.Orders o
JOIN ERP.Products p ON o.product_id = p.product_id
WHERE o.order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY)
GROUP BY p.product_id, p.name
ORDER BY 销售数量 DESC
LIMIT 10;`,
        reasoning: [
          '1. 识别时间范围: "过去30天" → WHERE order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY)',
          '2. 确定主要指标: "销量" → SUM(quantity)',
          '3. 关联表: Orders 表需要 JOIN Products 表获取产品名称',
          '4. 分组依据: 按产品分组 → GROUP BY product_id, name',
          '5. 排序和限制: Top10 → ORDER BY 销售数量 DESC LIMIT 10'
        ]
      };
    } else if (question.includes('月度')) {
      return {
        sql: `SELECT
  DATE_FORMAT(order_date, '%Y-%m') AS 月份,
  COUNT(*) AS 订单数,
  SUM(total_amount) AS 销售额,
  AVG(total_amount) AS 平均客单价
FROM ERP.Orders
WHERE order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 12 MONTH)
GROUP BY DATE_FORMAT(order_date, '%Y-%m')
ORDER BY 月份;`,
        reasoning: [
          '1. 识别时间范围: "月度" → 按月份分组',
          '2. 格式化日期: DATE_FORMAT(order_date, "%Y-%m") AS 月份',
          '3. 计算指标: 订单数(COUNT)、销售额(SUM)、平均客单价(AVG)',
          '4. 分组: GROUP BY 月份',
          '5. 时间筛选: WHERE order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 12 MONTH)'
        ]
      };
    } else {
      // 默认SQL
      return {
        sql: `SELECT * FROM ERP.Products LIMIT 100;`,
        reasoning: [
          '1. 查询产品表',
          '2. 限制返回100条记录'
        ]
      };
    }
  }
};