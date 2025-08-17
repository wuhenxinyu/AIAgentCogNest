"""
 LangChian 基础使用,思维链使用
"""
from langchain.prompts import PromptTemplate
from langchain_community.llms import Tongyi  # 导入通义千问Tongyi模型
import dashscope
import os
from langchain_openai import OpenAI

# 从环境变量获取 dashscope 的 API Key
os.environ['DASHSCOPE_API_KEY'] = 'sk-siRF78nIxVVBKekhvZF6POAzSrFXymXwzCFj4YT6SzFIlvWA'
# os.environ['DASHSCOPE_API_KEY'] = 'sk-58f051ae745e4bb19fdca31735105b11'
api_key = os.environ.get('DASHSCOPE_API_KEY')
# dashscope.api_key = api_key

# 加载 Tongyi 模型
# llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=api_key)  # 使用通义千问qwen-turbo模型


# 加载 OpenAI 模型
llm = OpenAI(
    temperature=0.8, 
    openai_api_key=api_key,
    model_name='qwen-turbo-latest',
    openai_api_base="https://api.fe8.cn/v1"
    ) 

# 创建Prompt Template
# 创建一个PromptTemplate对象
# PromptTemplate是LangChain中用于创建提示模板的类
# 它允许我们定义一个带有占位符的模板字符串，这些占位符可以在运行时被实际值替换
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

# 新推荐用法：将 prompt 和 llm 组合成一个"可运行序列"
chain = prompt | llm

# 使用 invoke 方法传入输入
result1 = chain.invoke({"product": "colorful socks"})
print(result1)

result2 = chain.invoke({"product": "广告设计"})
print(result2)

