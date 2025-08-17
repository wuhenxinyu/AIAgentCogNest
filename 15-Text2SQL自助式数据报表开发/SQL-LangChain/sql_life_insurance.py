"""
SQL + LLM 使用：
    • 通过SQLDatabase可以访问到数据库的Schema
    • agent_executor 作为SQL Agent，可以执行用户的各种SQL需求（通过自然语言 => 编写SQL => 查询结果返回）
    如果数据库中没有找到对应的表，会报OutputParseException错误
    如果有多张表，会分别执行，然后判断哪个数据表可以得到结果，最后将结果合并返回
"""
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase


db_user = "student123"
db_password = "student321"
db_host = "rm-uf6z891lon6dxuqblqo.mysql.rds.aliyuncs.com:3306"
db_name = "life_insurance"
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
#engine = create_engine('mysql+mysqlconnector://root:passw0rdcc4@localhost:3306/wucai')



from langchain.chat_models import ChatOpenAI
import os

# 从环境变量获取 dashscope 的 API Key
os.environ['DASHSCOPE_API_KEY'] = 'sk-siRF78nIxVVBKekhvZF6POAzSrFXymXwzCFj4YT6SzFIlvWA'
api_key = os.environ.get('DASHSCOPE_API_KEY')

# 初始化一个大语言模型（LLM）实例，使用 ChatOpenAI 类。
# ChatOpenAI 是 LangChain 库中用于与 OpenAI 聊天模型进行交互的类，
# 借助该类可以方便地调用大语言模型完成各种自然语言处理任务。
llm = ChatOpenAI(
    temperature=0.01,
    model="deepseek-v3",  
    openai_api_base = "https://api.fe8.cn/v1",
    openai_api_key  = api_key
)
# 需要设置llm
# 初始化 SQLDatabaseToolkit 实例，该工具包是 LangChain 中用于与 SQL 数据库交互的工具集合。
# 参数 db 传入之前创建的 SQLDatabase 实例，代表要操作的数据库。
# 参数 llm 传入之前初始化的大语言模型实例，用于理解自然语言并生成相应的 SQL 语句。
# 通过这个工具包，后续可以将自然语言查询转换为数据库可执行的 SQL 语句，实现与数据库的交互。
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True # 打印中间过程
)

# Task1
agent_executor.run("获取所有客户的姓名和联系电话")

agent_executor.run("找出所有已婚客户的保单。")

agent_executor.run("查询所有未支付保费的保单号和客户姓名。")
# 数据库中无该字段，返回查不到该字段；如："I don't know. The database does not have a table that contains all the required information for 姓名 (name), 联系电话 (phone), 身份证号 (id_card), and 地址 (address)."
agent_executor.run("查询所有客户的姓名、联系电话、身份证号、地址")

agent_executor.run("找出所有理赔金额大于10000元的理赔记录，并列出相关客户的姓名和联系电话。")

agent_executor.run("查找代理人的姓名和执照到期日期，按照执照到期日期升序排序。")

agent_executor.run("获取所有保险产品的产品名称和保费，按照保费降序排序。")

agent_executor.run("查询所有在特定销售区域工作的员工的姓名和职位。")

agent_executor.run("找出所有年龄在30岁以下的客户，并列出其客户ID、姓名和出生日期。")
agent_executor.run("查找所有已审核但尚未支付的理赔记录，包括理赔号、审核人和审核日期。")
agent_executor.run("获取每个产品类型下的平均保费，以及该产品类型下的产品数量。")