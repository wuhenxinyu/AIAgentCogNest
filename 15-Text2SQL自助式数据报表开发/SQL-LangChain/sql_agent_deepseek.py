"""
SQLDatabaseTookit 工具类，使用DeepSeek进行数据表的查询
采用LangChain框架:提供了sql chain，prompt，retriever，tools, agent，让用户通过自然语言，执行SQL查询
    方法1：SQLDatabase，提供了数据库的连接，执行SQL语句，获取结果。
    优点：使用方便，自动通过数据库连接，获取数据库的metadata
    不足：执行不灵活，需要多次判断哪个表适合复杂查询很难胜任，对于复杂查询通过率低。
    步骤：通过SQLDatabase可以访问到数据库的Schema，通过SQLDatabase可以访问到数据库的表数据;每一次操作都会对每一张表进行查看一遍，会对相同表进行多次查询。
    解决办法：给Agent配备专有知识库，在prompt中动态完善 和query相关的context，让Agent在执行SQL查询时，能够根据context，选择合适的表进行查询。
"""
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor

db_user = "student123"
db_password = "student321"
db_host = "rm-uf6z891lon6dxuqblqo.mysql.rds.aliyuncs.com:3306"
db_name = "action"
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")


from langchain.chat_models import ChatOpenAI
import os

# 从环境变量获取 dashscope 的 API Key
os.environ['DASHSCOPE_API_KEY'] = 'sk-siRF78nIxVVBKekhvZF6POAzSrFXymXwzCFj4YT6SzFIlvWA'
api_key = os.environ.get('DASHSCOPE_API_KEY')

# 通过LLM => 撰写SQL
llm = ChatOpenAI(
    temperature=0.01, # 温度，0.01 表示非常确定，1 表示非常不确定
    # model="deepseek-v3",  
    model = "qwen-turbo",
    openai_api_base = "https://api.fe8.cn/v1",
    openai_api_key  = api_key
)

# 需要设置llm
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# SQL智能体：给它目标，它自己会进行规划，最终把结果给你
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True # 打印中间过程
)


# Task: 描述数据表
agent_executor.run("描述与订单相关的表及其关系")

# 这个任务，实际上数据库中 没有HeroDetails表;如果没有找到对应的表，会解析报错OutputParseException
agent_executor.run("描述HeroDetails表")

# 这个任务，数据库中的表实际为 heros;也可以会找到多张表，逐一进行尝试
agent_executor.run("描述Hero表")

# 执行指定的任务：编写SQL => 查询结果; 如果多张同名的表会分别查询，发现哪个表可以找到答案就进行哪个表显示
agent_executor.run("找出英雄攻击力最高的前5个英雄")

