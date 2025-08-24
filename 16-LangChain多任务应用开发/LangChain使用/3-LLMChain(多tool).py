"""
  LangChain 多工具tools使用
"""
import os
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain_community.llms import Tongyi  # 导入通义千问Tongyi模型
from langchain.agents import AgentType
import dashscope
# 你需要在环境变量中添加 OPENAI_API_KEY 和 SERPAPI_API_KEY
os.environ['DASHSCOPE_API_KEY'] = 'sk-siRF78nIxVVBKekhvZF6POAzSrFXymXwzCFj4YT6SzFIlvWA'
os.environ["SERPAPI_API_KEY"] = '*******'
 
# 从环境变量获取 dashscope 的 API Key
api_key = os.environ.get('DASHSCOPE_API_KEY')
dashscope.api_key = api_key

# 加载 Tongyi 模型
llm = Tongyi(model_name="deepseek-v3", dashscope_api_key=api_key)  # 使用通义千问qwen-turbo模型


# 加载 serpapi, llm-math工具, 因为llm-math要使用LLM，所以后面需要指定LLM，使用多个Tools
tools = load_tools(["serpapi", "llm-math"], llm=llm)
 
"""
agent：代理类型  
    zero-shot-react-description: 根据工具的描述和请求内容的来决定使用哪个工具（最常用）
    react-docstore: 使用 ReAct 框架和 docstore 交互, 使用Search 和Lookup 工具, 前者用来搜, 后者寻找term, 举例: Wipipedia 工具
    self-ask-with-search 此代理只使用一个工具: Intermediate Answer, 它会为问题寻找事实答案(指的非 gpt 生成的答案, 而是在网络中,文本中已存在的), 如 Google search API 工具
    conversational-react-description: 为会话设置而设计的代理, 它的prompt会被设计的具有会话性, 且还是会使用 ReAct 框架来决定使用来个工具, 并且将过往的会话交互存入内存
"""
# 工具加载后需要初始化，verbose=True 代表打印执行详情
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
 
# 运行 agent
result = agent.invoke({"input": "当前北京的温度是多少华氏度？这个温度的1/4是多少"})
print(result)


#help(load_tools)
import langchain_community
help(langchain_community.llms)

