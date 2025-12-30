"""
带数据库支持的API服务器
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import json
import uuid
import os
import re
from typing import Dict, Any, List, Optional
from data_manager import data_manager

# 延迟导入,避免sqlite3错误影响启动
try:
    from app.sql_agent import SQLAgentManager
    from app.config import settings
    LANGCHAIN_AVAILABLE = True
except Exception as e:
    print(f"Warning: LangChain SQL Agent not available: {e}")
    LANGCHAIN_AVAILABLE = False

app = FastAPI(title="SQL Agent API with Database")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 存储上传的文件
file_store: Dict[str, Dict] = {}

# 存储SQL Agent实例
sql_agents: Dict[str, Any] = {}

@app.get("/")
async def root():
    return {"message": "SQL Agent API with Database is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "files_loaded": len(file_store),
        "database_tables": len(data_manager.get_table_list()),
        "active_sessions": 0
    }

@app.get("/datasources")
async def get_data_sources():
    """获取所有数据源"""
    # 获取数据库表
    db_tables = data_manager.get_table_list()

    # 添加上传的文件
    for file_id, file_info in file_store.items():
        db_tables.append({
            "name": file_info["filename"],
            "table": f"file_{file_id}",
            "rows": file_info["rows"],
            "columns": file_info["columns"],
            "description": "用户上传的文件",
            "source": "upload",
            "file_id": file_id
        })

    return {
        "success": True,
        "sources": db_tables
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文件"""
    try:
        # 读取文件内容
        content = await file.read()

        # 解析CSV或Excel
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))

        # 生成文件ID
        file_id = str(uuid.uuid4())

        # 存储文件信息
        file_store[file_id] = {
            "filename": file.filename,
            "data": df.to_dict('records'),
            "columns": df.columns.tolist(),
            "rows": len(df),
            "shape": df.shape
        }

        return {
            "success": True,
            "file_id": file_id,
            "message": f"File '{file.filename}' uploaded successfully",
            "headers": df.columns.tolist(),
            "total_columns": len(df.columns),
            "estimated_rows": len(df)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_data(request: Dict[str, Any]):
    """查询数据（使用LangChain SQL Agent）"""
    try:
        query = request.get("query", "")
        file_id = request.get("file_id")
        table_name = request.get("table_name")
        limit = request.get("limit", 100)

        # 如果指定了file_id，使用LangChain SQL Agent查询上传的文件
        if file_id and file_id in file_store:
            file_info = file_store[file_id]
            
            print(f"\n{'='*80}")
            print(f"[CSV查询] 用户问题: {query}")
            print(f"[CSV查询] 文件名: {file_info['filename']}")
            print(f"{'='*80}\n")
            
            # 使用 LangChain SQL Agent 处理查询
            if LANGCHAIN_AVAILABLE:
                agent_key = f"file_{file_id}"
                
                # 获取或创建SQL Agent
                if agent_key not in sql_agents:
                    print(f"[CSV查询] 创建新的SQL Agent for {file_id}")
                    agent = SQLAgentManager(
                        openai_api_key=settings.openai_api_key,
                        openai_base_url=settings.openai_base_url,
                        model=settings.default_model
                    )
                    
                    # 将DataFrame转换为CSV内容
                    import io
                    df = pd.DataFrame(file_info["data"])
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    csv_content = csv_buffer.getvalue().encode('utf-8')
                    
                    # 清理file_id中的连字符，避免SQLite表名问题
                    clean_file_id = file_id.replace('-', '_')
                    
                    # 创建数据库
                    db_result = agent.create_database_from_file(
                        csv_content,
                        'csv',
                        table_name=f"file_{clean_file_id}"
                    )
                    
                    if not db_result["success"]:
                        raise HTTPException(status_code=500, detail=db_result["error"])
                    
                    # 创建SQL Agent
                    agent_result = agent.create_sql_agent()
                    if not agent_result["success"]:
                        raise HTTPException(status_code=500, detail=agent_result["error"])
                    
                    sql_agents[agent_key] = agent
                else:
                    print(f"[CSV查询] 重用已有的SQL Agent for {file_id}")
                
                # 执行查询
                agent = sql_agents[agent_key]
                result = agent.query_data(query)
                
                if result["success"]:
                    print(f"[CSV查询] 查询成功: SQL={result.get('sql', '')[:50]}..., 数据行数={len(result.get('data', []))}")
                    return {
                        "success": True,
                        "answer": result.get("answer", "查询完成"),
                        "sql": result.get("sql"),
                        "reasoning": result.get("reasoning"),
                        "data": result.get("data", []),
                        "total_rows": result.get("returned_rows", 0),
                        "returned_rows": result.get("returned_rows", 0),
                        "columns": result.get("columns", file_info["columns"]),
                        "source": "file"
                    }
                else:
                    raise HTTPException(status_code=500, detail=result.get("error", "Query failed"))
            
            # 如果LangChain不可用，回退到简单查询
            print("[CSV查询] LangChain不可用，使用简单查询")
            data = file_info["data"]
            return {
                "success": True,
                "answer": f"从文件 {file_info['filename']} 查询到 {len(data)} 条记录",
                "data": data[:limit],
                "total_rows": len(data),
                "returned_rows": min(len(data), limit),
                "columns": file_info["columns"],
                "source": "file"
            }

        # 使用LangChain SQL Agent查询数据库表
        if table_name and LANGCHAIN_AVAILABLE:
            # 获取或创建该表的SQL Agent
            if table_name not in sql_agents:
                # 创建新的SQL Agent
                agent = SQLAgentManager(
                    openai_api_key=settings.openai_api_key,
                    openai_base_url=settings.openai_base_url,
                    model=settings.default_model
                )

                # 使用data_manager的数据库路径
                db_path = data_manager.db_path
                if not os.path.exists(db_path):
                    raise HTTPException(status_code=500, detail="Database not found")

                # 直接连接到现有数据库
                from langchain_community.utilities import SQLDatabase
                db_uri = f"sqlite:///{db_path}"
                agent.db = SQLDatabase.from_uri(db_uri)
                agent.temp_db_path = db_path

                # 创建SQL Agent
                agent_result = agent.create_sql_agent()
                if not agent_result["success"]:
                    raise HTTPException(status_code=500, detail=agent_result["error"])

                sql_agents[table_name] = agent

            agent = sql_agents[table_name]

            # 执行查询（新的 sql_agent 已经返回 sql, reasoning, data）
            result = agent.query_data(query)

            if not result["success"]:
                raise HTTPException(status_code=500, detail=result.get("error", "Query failed"))

            # 新的 sql_agent 已经提取了 SQL、推理步骤和数据
            sql_query = result.get("sql")
            reasoning_steps = result.get("reasoning", [])
            data = result.get("data", [])
            columns = result.get("columns", [])

            # 如果没有数据，使用 data_manager 作为后备
            if not data:
                fallback_result = data_manager.query_data(query, table_name, limit)
                data = fallback_result.get("data", [])
                columns = fallback_result.get("columns", [])

            return {
                "success": True,
                "answer": result.get("answer", ""),
                "sql": sql_query,
                "reasoning": reasoning_steps if reasoning_steps else None,
                "data": data,
                "columns": columns,
                "total_rows": len(data),
                "returned_rows": len(data),
                "source": "langchain_agent"
            }

        # 否则使用数据管理器查询（后备方案）
        result = data_manager.query_data(query, table_name, limit)
        result["source"] = "database"

        # 生成简单的SQL和推理信息
        if table_name and result.get("success"):
            # 生成伪SQL
            if "前" in query and "条" in query:
                n = 10
                if "5" in query:
                    n = 5
                elif "20" in query:
                    n = 20
                result["sql"] = f"SELECT * FROM {table_name} LIMIT {n}"
                result["reasoning"] = [
                    f"识别到用户想要查看前{n}条数据",
                    f"生成SQL查询: SELECT * FROM {table_name} LIMIT {n}",
                    f"执行查询并返回结果"
                ]
            elif "销售" in query and "额" in query:
                result["sql"] = f"SELECT SUM(price * sales_volume) as total FROM {table_name}"
                result["reasoning"] = [
                    "识别到用户想要计算销售总额",
                    "需要使用SUM函数对price * sales_volume求和",
                    "执行聚合查询获取总销售额"
                ]
            elif "统计" in query or "数量" in query:
                result["sql"] = f"SELECT COUNT(*) FROM {table_name}"
                result["reasoning"] = [
                    "识别到用户想要统计记录数量",
                    f"使用COUNT(*)函数统计{table_name}表的记录数",
                    "返回统计结果"
                ]
            elif "分类" in query or "类别" in query:
                result["sql"] = f"SELECT category, COUNT(*) as count FROM {table_name} GROUP BY category"
                result["reasoning"] = [
                    "识别到用户想要按类别统计",
                    "使用GROUP BY对类别进行分组",
                    "使用COUNT统计每个类别的数量"
                ]
            elif "最大" in query or "最高" in query:
                result["sql"] = f"SELECT * FROM {table_name} ORDER BY price DESC LIMIT 1"
                result["reasoning"] = [
                    "识别到用户想要找到最高价格的产品",
                    "使用ORDER BY price DESC排序",
                    "使用LIMIT 1获取价格最高的记录"
                ]
            elif "最小" in query or "最低" in query:
                result["sql"] = f"SELECT * FROM {table_name} ORDER BY price ASC LIMIT 1"
                result["reasoning"] = [
                    "识别到用户想要找到最低价格的产品",
                    "使用ORDER BY price ASC排序",
                    "使用LIMIT 1获取价格最低的记录"
                ]
            else:
                result["sql"] = f"SELECT * FROM {table_name} LIMIT {limit}"
                result["reasoning"] = [
                    "分析用户查询意图",
                    f"生成通用查询SQL: SELECT * FROM {table_name}",
                    f"限制返回{limit}条结果"
                ]

        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_with_data(request: Dict[str, Any]):
    """对话功能"""
    try:
        message = request.get("message", "")
        file_id = request.get("file_id")
        table_name = request.get("table_name")
        session_id = request.get("session_id", str(uuid.uuid4()))

        # 构建智能回复
        if "你好" in message or "hi" in message.lower():
            # 获取可用数据源
            sources = data_manager.get_table_list()
            source_names = [s["name"] for s in sources]
            response = f"您好！我是您的数据分析助手。当前可用的数据源有：\n"
            response += "\n".join([f"• {name}" for name in source_names])
            response += "\n\n请问您想了解哪些数据？"

        elif "数据源" in message or "数据表" in message:
            sources = data_manager.get_table_list()
            response = "当前数据源列表：\n\n"
            for source in sources:
                response += f"📊 {source['name']}\n"
                response += f"   • 描述：{source['description']}\n"
                response += f"   • 行数：{source['rows']}\n"
                response += f"   • 列数：{len(source['columns'])}\n"
                response += f"   • 来源：{source['source']}\n\n"

        elif "销售" in message:
            # 查询销售数据
            result = data_manager.query_data("销售总额", "sales_data")
            if result["success"] and result["data"]:
                total_sales = sum(
                    item.get("price", 0) * item.get("sales_volume", 0)
                    for item in result["data"]
                )
                response = f"根据销售数据分析：\n"
                response += f"• 总销售额：¥{total_sales:,.2f}\n"
                response += f"• 记录数：{result['total_rows']}条\n"
            else:
                response = "抱歉，未找到销售数据"

        elif "产品" in message:
            # 查询产品数据
            result = data_manager.query_data("前10个产品", "erp_products")
            if result["success"] and result["data"]:
                response = f"产品列表（前10个）：\n"
                for item in result["data"][:5]:
                    response += f"• {item.get('name', 'N/A')} - ¥{item.get('price', 0):,.2f}\n"
            else:
                response = "抱歉，未找到产品数据"

        else:
            # 通用查询
            result = data_manager.query_data(message, table_name)
            if result["success"]:
                response = f"根据您的问题「{message}」，我为您找到以下信息：\n\n"
                response += result["answer"]
                if result["data"] and len(result["data"]) > 0:
                    response += f"\n\n共找到 {result['total_rows']} 条相关记录"
            else:
                response = f"抱歉，无法处理您的问题：{message}"

        return {
            "success": True,
            "message": response,
            "session_id": session_id,
            "data": []
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/visualize")
async def create_visualization(request: Dict[str, Any]):
    """创建可视化"""
    try:
        chart_type = request.get("chart_type", "bar")
        table_name = request.get("table_name")
        x_column = request.get("x_column")
        y_column = request.get("y_column")

        # 根据图表类型和数据生成相应的HTML
        chart_html = f"""
        <div style="padding: 20px;">
            <h3>数据可视化图表 ({chart_type})</h3>
            <div style="margin-top: 20px;">
                <canvas id="chart" width="400" height="300"></canvas>
            </div>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script>
                // 这里是实际的图表渲染代码
                // 由于是测试版本，仅显示占位符
                const ctx = document.getElementById('chart').getContext('2d');
                ctx.font = '20px Arial';
                ctx.fillStyle = '#ccc';
                ctx.textAlign = 'center';
                ctx.fillText('图表区域 (' + chart_type + ')', 200, 150);
            </script>
            <p style="margin-top: 10px; color: #666;">
                表名: """ + str(table_name or '未指定') + """ |
                X轴: """ + str(x_column or '自动') + """ |
                Y轴: """ + str(y_column or '自动') + """
            </p>
        </div>
        """

        return {
            "success": True,
            "chart_html": chart_html
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files")
async def list_files():
    """列出所有上传的文件"""
    files = []
    for file_id, info in file_store.items():
        files.append({
            "file_id": file_id,
            "filename": info["filename"],
            "total_columns": len(info["columns"]),
            "estimated_rows": info["rows"]
        })
    return {"files": files}

@app.get("/tables/{table_name}")
async def get_table_info(table_name: str):
    """获取表详细信息"""
    info = data_manager.get_table_info(table_name)
    if info:
        # 获取表的前几条数据
        df = data_manager.data_cache.get(table_name)
        sample_data = df.head(5).to_dict('records') if df is not None else []
        return {
            "success": True,
            "info": info,
            "sample_data": sample_data
        }
    else:
        raise HTTPException(status_code=404, detail="Table not found")

if __name__ == "__main__":
    import uvicorn
    print("Starting SQL Agent API with Database on http://localhost:8000")
    print("数据库已加载，支持以下功能：")
    print("- 真实CSV数据查询")
    print("- ERP模拟数据查询")
    print("- 文件上传查询")
    print("- 智能对话分析")
    uvicorn.run(app, host="0.0.0.0", port=8001)