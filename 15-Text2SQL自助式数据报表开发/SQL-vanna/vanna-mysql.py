"""
pip install vanna，可选扩展如 vanna[chromadb,ollama,mysql] 支持本地化部署。
"""
from vanna.openai import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
import mysql.connector
import time
from openai import OpenAI

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    """
    MyVanna类继承自ChromaDB_VectorStore和OpenAI_Chat，用于实现自然语言到SQL的转换功能。
    
    主要功能:
    1. 使用ChromaDB存储和检索向量化的SQL查询和数据库模式信息
    2. 利用OpenAI的语言模型将自然语言转换为SQL查询
    3. 管理配置信息，确保ChromaDB和OpenAI组件正确初始化
    4. 提供连接到MySQL数据库的功能
    5. 执行SQL查询并返回结果
    6. 处理查询结果，返回DataFrame格式
    7. 提供自然语言到SQL的转换功能
    8. 提供SQL查询执行功能
    9. 提供查询结果处理功能
    """
    def __init__(self, config=None):
        # 分离配置，只将OpenAI客户端传递给OpenAI_Chat
        # 如果config参数存在,创建config的副本作为chroma_config;否则创建空字典
        # 这样做是为了避免直接修改原始config字典,保持数据隔离
        chroma_config = config.copy() if config else {}
        openai_config = config.copy() if config else {}
        
        # 保存client到实例属性
        if config and 'client' in config:
            self.client = config['client']
        else:
            self.client = None
        
        # 从chroma_config中移除client
        if 'client' in chroma_config:
            del chroma_config['client']
            
        # 初始化两个基类
        # 初始化ChromaDB向量存储基类,用于存储和检索向量化的SQL查询和模式信息
        ChromaDB_VectorStore.__init__(self, config=chroma_config)
        # 初始化OpenAI聊天基类,用于处理自然语言到SQL的转换和对话功能
        OpenAI_Chat.__init__(self, config=openai_config)

# 创建OpenAI客户端
client = OpenAI(
    api_key = 'sk-siRF78nIxVVBKekhvZF6POAzSrFXymXwzCFj4YT6SzFIlvWA' ,
    base_url='https://api.fe8.cn/v1'
)

# 初始化Vanna实例
vn = MyVanna(config={
    'model': 'deepseek-v3', 
    'client': client
})

# Vanna需要连接到MySQL数据库的原因:
# 1. 获取数据库表结构信息(schema)用于训练
# 2. 执行生成的SQL查询并获取实际数据
# 3. 验证生成的SQL语句的正确性
vn.connect_to_mysql(host='rm-uf6z891lon6dxuqblqo.mysql.rds.aliyuncs.com', 
                    dbname='action', user='student123', password='student321', port=3306)

# 连接到MySQL数据库
try:
    connection = mysql.connector.connect(
        host='rm-uf6z891lon6dxuqblqo.mysql.rds.aliyuncs.com',
        database='action',
        user='student123',
        password='student321',
        port=3306
    )
    print("成功连接到MySQL数据库")
    
    # 获取所有表名
    cursor = connection.cursor()
    cursor.execute("""
        SELECT TABLE_NAME 
        FROM information_schema.TABLES 
        WHERE TABLE_SCHEMA = 'action'
    """)
    tables = cursor.fetchall()
    
    # 训练每个表的schema
    for (table_name,) in tables:
        try:
            # 获取表的创建语句
            cursor.execute(f"SHOW CREATE TABLE {table_name}")
            _, create_table = cursor.fetchone()
            
            print(f"正在训练表 {table_name} 的schema...")
            # 训练模型，通过 DDL、文档或 SQL 示例训练
            vn.train(ddl=create_table)
            
        except Exception as e:
            print(f"训练表 {table_name} 失败: {str(e)}")
            continue
    
    print("Schema训练完成")
    
except mysql.connector.Error as err:
    print(f"数据库连接错误: {err}")
finally:
    if 'connection' in locals() and connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL连接已关闭")


 # 示例：使用Vanna进行自然语言查询
question = "找出英雄攻击力最高的前5个英雄"
#print(f"\n问题: {question}")
# 提问与查询
"""
ask函数
    作用：用户通过自然语言提问时调用此函数，它是查询的核心入口，会依次调用generate_sql、run_sql、generate_plotly_code、get_plotly_figure四个函数来完成整个查询及可视化的过程。
    工作流程：
        • 首先将用户的问题转换成向量表示，然后在向量数据库中检索与问题语义最相似的DDL语句、文档和SQL查询。
        • 将检索到的信息和用户的问题一起提供给LLM，生成对应的SQL查询。
        • 执行生成的SQL查询，并将查询结果以表格和Plotly图表的形式返回给用户。
"""
vn.ask(question)

# #help(vn.ask)
# #vn.ask("查询heros表中 英雄攻击力前5名的英雄")
"""
generate_sql函数
    作用：根据用户输入的自然语言问题，生成对应的SQL语句。
    工作流程：
    • 调用get_similar_question_sql函数，在向量数据库中检索与问题相似的sql/question对。
    • 调用get_related_ddl函数，在向量数据库中检索与问题相似的建表语句ddl。
    • 调用get_related_documentation函数，在向量数据库中检索与问题相似的文档。
    • 调用get_sql_prompt函数，结合上述检索到的信息生成prompt，然后将prompt提供给LLM，生成SQL语句。
"""
# sql=vn.generate_sql("查询heros表中 英雄攻击力前5名的英雄")
# print('sql=', sql)
"""
run_sql函数
    作用：执行generate_sql函数生成的SQL语句，并返回查询结果。
    工作流程：将生成的SQL语句发送到连接的数据库中执行，获取并返回查询结果。
"""
# df=vn.run_sql(sql)
# print('df=', df)
