"""
DeepSeek + Faiss搭建本地知识库检索ChatPDF-Faiss; MultiQueryRetriever使用
相似语义改写：使用大模型将用户查询改写成多个语义相近的查询，提升召回多样性。例如，LangChain的MultiQueryRetriever支持多查询召回，再进行回答问题
"""
from langchain.retrievers import MultiQueryRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
import os
os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here'
# 获取环境变量中的 DASHSCOPE_API_KEY
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")
llm = Tongyi(model_name="deepseek-v3", dashscope_api_key=DASHSCOPE_API_KEY) # qwen-turbo

# 创建嵌入模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY,
)

# 加载向量数据库，添加allow_dangerous_deserialization=True参数以允许反序列化
vectorstore = FAISS.load_local("./faiss-1", embeddings, allow_dangerous_deserialization=True)

# 创建MultiQueryRetriever
# MultiQueryRetriever是一个高级检索器，它可以:
# 1. 基于用户的原始查询，使用LLM生成多个不同的查询变体
# 2. 对每个查询变体执行检索
# 3. 合并所有检索结果并去重
retriever = MultiQueryRetriever.from_llm(
    # 使用向量数据库作为基础检索器
    retriever=vectorstore.as_retriever(),
    # 使用之前定义的LLM来生成查询变体
    llm=llm
)

# 示例查询
query = "客户经理的考核标准是什么？"
# 执行查询
# 使用MultiQueryRetriever检索器获取相关文档
# 这行代码会:
# 1. 基于原始查询生成多个语义相近的查询变体
# 2. 对每个查询变体执行检索
# 3. 合并并去重所有检索结果
# 4. 返回最终的相关文档列表
results = retriever.get_relevant_documents(query)

# 打印结果
print(f"查询: {query}")
print(f"找到 {len(results)} 个相关文档:")
for i, doc in enumerate(results):
    print(f"\n文档 {i+1}:")
    print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)