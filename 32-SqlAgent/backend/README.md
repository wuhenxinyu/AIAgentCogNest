# SQL Agent Backend

基于 FastAPI 和 LangChain 的 SQL 数据分析后端服务。

## 功能特性

- 📊 支持 CSV 和 Excel 文件上传
- 🤖 集成 LangChain SQL Agent 实现自然语言查询
- 📈 多种数据可视化图表（柱状图、折线图、饼图、散点图等）
- 💬 对话式数据分析
- 🗄️ 自动从文件创建 SQL 数据库
- 🔒 环境变量配置管理

## 项目结构

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI 主应用
│   ├── config.py        # 配置管理
│   ├── sql_agent.py     # LangChain SQL Agent
│   ├── visualization.py # 数据可视化
│   └── models.py        # Pydantic 模型
├── utils/
│   ├── __init__.py
│   └── file_processor.py # 文件处理工具
├── data/                # 数据存储目录
├── requirements.txt     # Python 依赖
├── run.py              # 启动脚本
└── README.md
```

## 安装和运行

### 1. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

### 2. 配置环境变量

编辑根目录的 `.env` 文件：

```env
# API Keys
OPENAI_API_KEY=your_openai_key_here

# FastAPI Configuration
HOST=0.0.0.0
PORT=8001
DEBUG=true

# 其他配置...
```

### 3. 运行服务器

```bash
python run.py
```

或使用 uvicorn：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

## API 端点

### 文件上传

```http
POST /upload
Content-Type: multipart/form-data

file: [CSV或Excel文件]
```

### 自然语言查询

```http
POST /query
Content-Type: application/json

{
    "query": "显示销售额最高的前10个产品",
    "file_id": "file-uuid",
    "limit": 100
}
```

### 数据可视化

```http
POST /visualize
Content-Type: application/json

{
    "file_id": "file-uuid",
    "chart_type": "bar",
    "x_column": "product_name",
    "y_column": "sales",
    "title": "产品销售额"
}
```

### 对话分析

```http
POST /chat
Content-Type: application/json

{
    "message": "分析这个数据的趋势",
    "file_id": "file-uuid",
    "session_id": "session-uuid"  # 可选
}
```

## 技术栈

- **FastAPI**: 高性能 Web 框架
- **LangChain**: LLM 应用开发框架
- **Pandas**: 数据处理
- **SQLite**: 轻量级数据库
- **Plotly**: 交互式可视化
- **Matplotlib/Seaborn**: 静态图表
- **Pydantic**: 数据验证
- **Python-dotenv**: 环境变量管理

## 使用说明

1. 首先上传 CSV 或 Excel 文件获取 `file_id`
2. 使用 `file_id` 进行自然语言查询或创建可视化
3. 支持连续对话，系统会记住上下文

## 注意事项

- 确保设置有效的 `OPENAI_API_KEY`
- 大文件处理可能需要较长时间
- 建议使用 CSV 格式以获得更好的性能
- 查询结果会自动限制数量以避免性能问题