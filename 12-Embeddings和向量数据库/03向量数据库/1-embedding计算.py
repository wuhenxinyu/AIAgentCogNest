"""
计算文本的embedding向量

Thinking：Embedding模型的核心特性是什么？
    1、Embedding模型将文本等离散数据转换为低维、稠密的向量，捕捉其语义信息。
    2、向量空间中的距离（如余弦相似度）可反映文本间的语义相似度。

动态调整维度是Jina-embedding模型赋予开发者的一个强大选项
Embedding模型的选择属于综合评估，即结合测试结果、模型的推理速度、部署成本 => 做出最终决策

多语言Embedding的优势是能将不同语言的文本映射到统一的语义空间。“clean room”、“部屋が綺麗”和“干净的房间”的向量在空间中会非常接近。=> 跨语言的聚类分析和检索才能实现。如：如 m3e-base 或 multilingual-e5-large

向量数据库的核心价值？
• 为大模型提供长期记忆： 弥补LLM上下文窗口（Context Window）长度限制和知识更新延迟的问题。
• 实现私有知识库的问答与搜索： 将企业内部文档、产品信息等转化为向量，实现基于语义的智能检索。
• 赋能推荐系统、以图搜图等多种应用： 通过计算用户、物品的向量相似度，提供更精准的推荐。
"""
import os
from openai import OpenAI
os.environ["DASHSCOPE_API_KEY"] = "your_api_key_here"

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
)

completion = client.embeddings.create(
    model="text-embedding-v4",
    input='我想知道迪士尼的退票政策',
    dimensions=1024, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
    encoding_format="float"
)

print(completion.model_dump_json())