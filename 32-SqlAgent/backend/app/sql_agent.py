import pandas as pd
import tempfile
import os
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent  # 新的 API！
from sqlalchemy import create_engine, text
import logging

logger = logging.getLogger(__name__)


class SQLAgentManager:
    """管理LangChain SQL Agent的创建和执行"""

    def __init__(self, openai_api_key: Optional[str] = None, openai_base_url: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        初始化SQL Agent管理器

        Args:
            openai_api_key: OpenAI API密钥
            openai_base_url: OpenAI Base URL
            model: 使用的模型名称
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_base_url = openai_base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model
        self.llm = None
        self.agent_executor = None
        self.db_connection = None
        self.temp_db_path = None

        if self.openai_api_key:
            self._initialize_llm()

    def _initialize_llm(self):
        """初始化LLM"""
        try:
            # 构建ChatOpenAI参数
            kwargs = {
                "model": self.model,
                "temperature": 0.0,
                "api_key": self.openai_api_key
            }

            # 如果设置了自定义base_url，添加到参数中
            if self.openai_base_url:
                kwargs["base_url"] = self.openai_base_url

            self.llm = ChatOpenAI(**kwargs)
            logger.info(f"LLM initialized successfully with model: {self.model}")
            if self.openai_base_url:
                logger.info(f"Using custom base URL: {self.openai_base_url}")
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")

    def create_database_from_file(self, file_content: bytes, file_type: str,
                                 table_name: str = "data_table") -> Dict[str, Any]:
        """
        从文件创建SQLite数据库

        Args:
            file_content: 文件内容
            file_type: 文件类型 ('csv' 或 'excel')
            table_name: 表名

        Returns:
            创建结果
        """
        try:
            # 创建临时数据库
            temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
            self.temp_db_path = temp_file.name
            temp_file.close()

            # 读取文件数据
            if file_type == 'csv':
                df = pd.read_csv(pd.io.common.BytesIO(file_content))
            elif file_type in ['excel', 'xlsx', 'xls']:
                df = pd.read_excel(pd.io.common.BytesIO(file_content))
            else:
                return {"success": False, "error": f"Unsupported file type: {file_type}"}

            # 清理列名（确保是有效的SQL标识符）
            df.columns = [self._clean_column_name(col) for col in df.columns]

            # 使用SQLAlchemy创建连接
            db_uri = f"sqlite:///{self.temp_db_path}"
            engine = create_engine(db_uri)
            self.db_connection = engine

            # 将数据写入数据库
            df.to_sql(table_name, engine, if_exists='replace', index=False)

            # 创建SQLDatabase对象
            self.db = SQLDatabase.from_uri(db_uri)

            logger.info(f"Database created successfully with table '{table_name}'")

            return {
                "success": True,
                "table_name": table_name,
                "rows": len(df),
                "columns": df.columns.tolist(),
                "db_path": self.temp_db_path
            }

        except Exception as e:
            logger.error(f"Error creating database: {str(e)}")
            return {"success": False, "error": str(e)}

    def _clean_column_name(self, col_name: str) -> str:
        """清理列名以符合SQL标识符规范"""
        # 移除特殊字符，替换为下划线
        cleaned = "".join(c if c.isalnum() or c == '_' else '_' for c in str(col_name))
        # 确保不以数字开头
        if cleaned and cleaned[0].isdigit():
            cleaned = "col_" + cleaned
        # 确保不为空
        if not cleaned:
            cleaned = "unnamed_column"
        return cleaned

    def create_sql_agent(self, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        创建SQL Agent（使用新的 create_agent API）

        Args:
            system_prompt: 自定义系统提示

        Returns:
            创建结果
        """
        try:
            if not self.llm:
                return {"success": False, "error": "LLM not initialized"}

            if not hasattr(self, 'db'):
                return {"success": False, "error": "Database not created"}

            # 默认系统提示
            default_system_prompt = f"""你是一个专业的数据分析师，专门帮助用户查询和分析 {self.db.dialect} 数据库。

你有以下工具可以使用：
- sql_db_list_tables: 列出数据库中的所有表
- sql_db_schema: 查看特定表的结构和示例数据
- sql_db_query: 执行 SQL 查询并返回结果
- sql_db_query_checker: 在执行前检查 SQL 查询的正确性

**执行步骤：**
1. **重要**: 使用 sql_db_list_tables 查看数据库中实际的表名（绝对不要猜测表名或使用 "table" 作为表名）
2. 使用 sql_db_schema 查看表的结构和列名
3. 仔细理解用户问题，提取关键信息：
   - 如果用户要求"前N条"、"显示N条"、"N个"，SQL 必须使用 LIMIT N
   - 如果用户没有指定数量，默认使用 LIMIT 10
   - 如果用户要求"所有"、"全部"，可以不加 LIMIT 或使用较大值
4. 根据步骤1和2查到的实际表名和列名，生成准确的 SQL 查询
5. 使用 sql_db_query_checker 检查 SQL 正确性
6. 使用 sql_db_query 执行查询

**重要约束：**
- 只使用 SELECT 语句，禁止 INSERT/UPDATE/DELETE
- **必须先调用 sql_db_list_tables 查看实际表名，绝对不要使用 "table" 或猜测的表名**
- **使用查询到的真实表名编写SQL（例如：file_xxx）**
- 必须根据用户指定的数量生成 LIMIT 子句
- 如果出错，分析错误并重新生成 SQL
- 绝对不要忽略用户在问题中指定的数量要求

**输出格式：**
查询完成后，请根据用户的具体问题，用 Markdown 格式输出针对性的数据分析报告，包含：

## 📊 数据分析报告

### 核心发现
- 针对用户问题，总结最重要的 2-3 个发现（用具体数据支撑，引用查询结果中的关键数字）

### 详细分析
- 深入分析查询结果，回答用户的问题
- 分析数据的分布、趋势或特征（如果查询涉及）
- 指出异常值或有趣的模式（如果存在）
- 对比不同维度的数据（如果查询涉及对比）

### 建议
- 基于查询结果和用户问题，给出 1-2 条可执行的建议

**重要提示：**
1. 分析报告必须直接回答用户的问题，不要使用通用模板
2. 引用查询结果中的具体数据（如：销售额最高的产品是XX，金额为XX元）
3. 不要在报告中列出原始数据，数据会自动显示在表格中
4. 不要在报告中提及文件名或记录总数，专注于分析查询结果
5. 如果用户只是要求显示数据（如"显示前10条"），简要总结查询到的数据特征即可，不需要冗长分析

**示例：**
- 用户问："查询前10个数据" → 
  步骤1: 调用 sql_db_list_tables 得到表名 "file_abc123"
  步骤2: SQL: SELECT * FROM file_abc123 LIMIT 10 
  步骤3: 报告：简要描述这10条数据的主要特征
  
- 用户问："销售额最高的前5个产品" → 
  步骤1: 调用 sql_db_list_tables 得到表名
  步骤2: SQL: SELECT * FROM <实际表名> ORDER BY sales DESC LIMIT 5 
  步骤3: 报告：列出TOP5产品及其销售额，并分析"""

            prompt = system_prompt or default_system_prompt

            # 创建 SQL 工具包
            toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
            tools = toolkit.get_tools()

            logger.info(f"创建 SQL Agent，可用工具: {[tool.name for tool in tools]}")

            # 使用新的 create_agent API（不会触发 transformers 依赖）
            self.agent_executor = create_agent(
                model=self.llm,
                tools=tools,
                system_prompt=prompt
            )

            return {"success": True, "message": "SQL Agent created successfully"}

        except Exception as e:
            logger.error(f"Error creating SQL agent: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def query_data(self, question: str) -> Dict[str, Any]:
        """
        使用SQL Agent查询数据

        Args:
            question: 自然语言查询问题

        Returns:
            查询结果
        """
        try:
            if not self.agent_executor:
                return {"success": False, "error": "SQL Agent not created"}

            # 使用新的 invoke 格式
            result = self.agent_executor.invoke({
                "messages": [{"role": "user", "content": question}]
            })

            # 从返回的 messages 中提取最后一条（agent 的回复）
            messages = result.get("messages", [])
            
            # 调试日志
            logger.info(f"Agent 返回了 {len(messages)} 条消息")
            for i, msg in enumerate(messages):
                logger.info(f"消息 {i}: type={type(msg).__name__}, has_tool_calls={hasattr(msg, 'tool_calls')}")
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    logger.info(f"  工具调用数量: {len(msg.tool_calls)}")
                    for j, tc in enumerate(msg.tool_calls):
                        logger.info(f"  工具调用 {j}: {tc}")
            
            # 找到最后一条 AI 消息（包含完整分析报告）
            answer = ""
            for msg in reversed(messages):
                if hasattr(msg, 'content') and msg.content:
                    content = msg.content
                    # 如果内容包含 Markdown 格式的分析报告，直接使用
                    if '## 📊' in content or '数据分析报告' in content or '核心发现' in content:
                        answer = content
                        break
                    # 否则累积所有有意义的内容
                    if content.strip() and content.strip() not in ['查询完成', '已找到', '查询成功']:
                        answer = content
                        break
            
            # 如果没有找到合适的答案，尝试组合所有消息
            if not answer or len(answer) < 50:
                all_contents = []
                for msg in reversed(messages):
                    if hasattr(msg, 'content') and msg.content:
                        content = msg.content.strip()
                        if content and content not in ['查询完成', '已找到', '查询成功']:
                            all_contents.append(content)
                if all_contents:
                    answer = '\n\n'.join(all_contents)
            
            # 如果还是没有，使用默认提示
            if not answer or len(answer) < 20:
                answer = "查询完成，请查看下方数据表格和分析图表。"
            
            # 提取工具调用和 SQL
            sql_queries = []
            reasoning_steps = []
            
            for msg in messages:
                # 检查是否有工具调用
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        # tool_call 可能是字典或对象
                        if isinstance(tool_call, dict):
                            tool_name = tool_call.get('name', '')
                            tool_args = tool_call.get('args', {})
                        else:
                            # 如果是对象，使用属性访问
                            tool_name = getattr(tool_call, 'name', '')
                            tool_args = getattr(tool_call, 'args', {})
                        
                        reasoning_steps.append(f"调用工具: {tool_name}")
                        
                        # 提取 SQL 查询
                        if tool_name == 'sql_db_query':
                            if isinstance(tool_args, dict):
                                sql = tool_args.get('query', '')
                            else:
                                sql = getattr(tool_args, 'query', '')
                            
                            if sql:
                                # 清理 SQL 中可能的 HTML/Tailwind 标记
                                import re
                                sql_clean = sql
                                # 移除各种可能的 HTML 标记
                                sql_clean = re.sub(r'\d+\s+font-[a-z-]+["\']?>', '', sql_clean)
                                sql_clean = re.sub(r'<[^>]+>', '', sql_clean)  # 移除所有 HTML 标签
                                sql_clean = re.sub(r'className="[^"]*"', '', sql_clean)  # 移除 className
                                sql_clean = sql_clean.strip()
                                
                                if sql_clean:
                                    sql_queries.append(sql_clean)
                                    logger.info(f"提取到 SQL (长度 {len(sql_clean)}): {sql_clean[:100]}...")
                                reasoning_steps.append(f"执行 SQL 查询")
                        elif tool_name == 'sql_db_schema':
                            if isinstance(tool_args, dict):
                                tables = tool_args.get('table_names', '')
                            else:
                                tables = getattr(tool_args, 'table_names', '')
                            reasoning_steps.append(f"查看表结构: {tables}")
                        elif tool_name == 'sql_db_list_tables':
                            reasoning_steps.append("列出所有数据库表")
                        elif tool_name == 'sql_db_query_checker':
                            reasoning_steps.append("检查 SQL 语法正确性")

            # 获取最后一个SQL查询
            sql = sql_queries[-1] if sql_queries else None
            
            # 如果没有提取到推理步骤，添加默认步骤
            if not reasoning_steps:
                reasoning_steps = [
                    f"分析问题: {question}",
                    "查询数据库并生成答案"
                ]

            # 提取实际的查询数据
            data = []
            columns = []
            if sql:
                try:
                    # 执行 SQL 获取实际数据
                    sql_result = self.execute_custom_sql(sql)
                    if sql_result["success"]:
                        data = sql_result["data"]
                        columns = sql_result["columns"]
                        logger.info(f"成功执行 SQL，返回 {len(data)} 行数据")
                except Exception as e:
                    logger.warning(f"执行 SQL 获取数据失败: {e}")

            return {
                "success": True,
                "answer": answer or "查询完成",
                "sql": sql,
                "reasoning": reasoning_steps,
                "data": data,
                "columns": columns,
                "returned_rows": len(data)
            }

        except Exception as e:
            logger.error(f"Error querying data: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def get_table_schema(self) -> Dict[str, Any]:
        """
        获取数据库表结构信息

        Returns:
            表结构信息
        """
        try:
            if not hasattr(self, 'db'):
                return {"success": False, "error": "Database not created"}

            # 获取所有表
            tables = self.db.get_usable_table_names()

            schema_info = {}
            for table in tables:
                # 获取表结构
                schema = self.db.get_table_info(table_names=[table])
                schema_info[table] = schema

            return {
                "success": True,
                "tables": tables,
                "schema": schema_info
            }

        except Exception as e:
            logger.error(f"Error getting table schema: {str(e)}")
            return {"success": False, "error": str(e)}

    def execute_custom_sql(self, sql_query: str) -> Dict[str, Any]:
        """
        执行自定义SQL查询

        Args:
            sql_query: SQL查询语句

        Returns:
            查询结果
        """
        try:
            if not self.db_connection:
                return {"success": False, "error": "Database connection not established"}

            # 使用SQLAlchemy执行查询
            with self.db_connection.connect() as conn:
                result = conn.execute(text(sql_query))

                # 获取列名
                columns = list(result.keys())

                # 获取数据
                rows = result.fetchall()

                # 转换为字典列表
                data = [dict(zip(columns, row)) for row in rows]

            return {
                "success": True,
                "data": data,
                "columns": columns,
                "row_count": len(data)
            }

        except Exception as e:
            logger.error(f"Error executing SQL: {str(e)}")
            return {"success": False, "error": str(e)}

    def cleanup(self):
        """清理临时文件"""
        try:
            if self.db_connection:
                self.db_connection.dispose()
            if self.temp_db_path and os.path.exists(self.temp_db_path):
                os.unlink(self.temp_db_path)
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")