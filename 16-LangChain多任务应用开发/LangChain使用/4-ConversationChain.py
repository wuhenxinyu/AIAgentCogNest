"""
    LangChain Memory 短期记忆使用：
    • Chains 和 Agent之前是无状态的，如果你想让他能记住之前的交互，就需要引入 内存
    • 可以让LLM拥有短期记忆
    • 对话过程中，记住用户的input 和 中间的output
"""
import os
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain_community.llms import Tongyi  # 导入通义千问Tongyi模型
from langchain.agents import AgentType
from langchain import ConversationChain
import dashscope

# 从环境变量获取 dashscope 的 API Key
os.environ['DASHSCOPE_API_KEY'] = 'sk-58f051ae745e4bb19fdca31735105b11'
api_key = os.environ.get('DASHSCOPE_API_KEY')
dashscope.api_key = api_key


# 使用通义千问Tongyi类加载模型（推荐）
# 如果您使用的是通义千问模型或dashscope密钥，请使用此方案
llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=api_key)  # 使用通义千问qwen-turbo模型

# 使用带有memory的ConversationChain
"""
在LangChain中提供了几种短期记忆的方式：
• BufferMemory: 将之前的对话完全存储下来，传给LLM
• BufferWindowMemory: 最近的K组对话存储下来，传给LLM
• ConversionMemory: 对对话进行摘要，将摘要存储在内存中，相当于将压缩过的历史对话传递给LLM
• VectorStore-backed Memory: 将之前所有对话通过向量存储到VectorDB（向量数据库）中，每次对话，会根据用户的输入信息，匹配向量数据库中最相似的K组对话
"""
conversation = ConversationChain(llm=llm, verbose=True)

# 使用 ConversationChain 实例的 predict 方法与大语言模型进行交互。
# 传入的 input 参数为 "Hi there!"，表示向模型发送的第一条消息。
# 模型会根据当前的对话历史（此时为空）生成对应的回复，
# 并将生成的回复存储在 output 变量中。
output = conversation.predict(input="Hi there!")
print(output)



output = conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
print(output)

