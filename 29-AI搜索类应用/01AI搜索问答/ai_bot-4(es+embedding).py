"""
RAG使用Elasticsearch检索文档 - 基于 embedding向量的召回策略 (优化版)
   使用更小的chunk大小来提升检索精度
   用 elasticsearch来进行文档片段chunks进行索引和管理，提升检索性能。
   基于 embedding 向量实现，使用 Dashscope 的 text-embedding-v4 模型计算向量相似度，实现语义搜索。
   优化版：
       1. 减少文档分块大小（chunk_size）为500，以提高检索精度。
       2. 增加chunk_overlap为50，以保持上下文连续性。
"""
import os
import asyncio
from typing import Optional
from .agents import Assistant
from .gui import WebUI
import warnings
from elasticsearch import Elasticsearch
from pypdf import PdfReader
from openai import OpenAI
import logging

warnings.filterwarnings("ignore")

# 设置日志级别为INFO，确保日志能够输出
logging.basicConfig(level=logging.INFO)

# 从环境变量获取 API Key，如果未设置则使用默认值
api_key = os.getenv('DASHSCOPE_API_KEY', 'your_api_key_here')
os.environ['DASHSCOPE_API_KEY'] = api_key

def get_embedding(text: str, client: OpenAI) -> list:
    """使用 Dashscope 的 text-embedding-v4 模型为文本生成向量。"""
    try:
        # 确保文本不为空
        if not text.strip():
            return []
        
        response = client.embeddings.create(
            model="text-embedding-v4",
            input=text,
            dimensions=1024,  # Qwen3-Embedding 系列支持的维度
            encoding_format="float"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"  - 获取 embedding 时出错: {e}")
        return []


def chunk_text(text: str, chunk_size: int = 4000, overlap: int = 200) -> list:
    """将长文本切分成带有重叠的块。"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def init_agent_service():
    """初始化具备 Elasticsearch + Embedding RAG 能力的助手服务"""

    # 步骤 1: 初始化 Elasticsearch 和 embedding 客户端
    try:
        es_client = Elasticsearch(
            "http://localhost:9200",
            # basic_auth=("elastic", "euqPcOlHrmW18rtaS-3P"),
            verify_certs=False
        )
        if not es_client.ping():
            print("无法连接到 Elasticsearch。")
            return None
        print("成功连接到 Elasticsearch！")
    except Exception as e:
        print(f"ES 连接时发生错误: {e}")
        return None

    # 初始化 Dashscope (OpenAI-compatible) 客户端
    try:
        embedding_client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        print("成功连接到 embedding 客户端！")
    except Exception as e:
        print(f"初始化 embedding 客户端时发生错误: {e}")
        return None

    # 步骤 2: 创建一个支持向量搜索的索引
    index_name = "my_insurance_docs_embedding_index"
    if es_client.indices.exists(index=index_name):
        print(f"索引 '{index_name}' 已存在，正在删除旧索引...")
        es_client.indices.delete(index=index_name)

    print(f"正在创建新索引 '{index_name}'...")
    es_client.indices.create(
        index=index_name,
        mappings={
            "properties": {
                "file_name": {"type": "keyword"},
                "content": {"type": "text"},
                "content_vector": {
                    "type": "dense_vector",
                    "dims": 1024,  # 必须与 embedding 模型的维度一致
                    "index": "true",
                    "similarity": "cosine" # 使用余弦相似度
                }
            }
        }
    )

    # 步骤 3: 索引文档
    docs_folder = 'docs'
    if not os.path.exists(docs_folder):
        print(f"错误：未找到 '{docs_folder}' 文件夹。")
        return None

    print("\n正在从 'docs' 文件夹读取、分块、生成向量并索引文档...")
    doc_id_counter = 1
    for filename in os.listdir(docs_folder):
        file_path = os.path.join(docs_folder, filename)
        if os.path.isfile(file_path):
            try:
                content = ""
                if filename.lower().endswith('.pdf'):
                    reader = PdfReader(file_path)
                    for page in reader.pages:
                        content += page.extract_text() or ""
                elif filename.lower().endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                else:
                    continue
                
                if content.strip():
                    # 将长文本分块，使用较小的chunk_size以提高检索精度
                    text_chunks = chunk_text(content, chunk_size=500, overlap=50)
                    print(f"  - 文件 '{filename}'被切分成 {len(text_chunks)} 个块。")
                    
                    for i, chunk in enumerate(text_chunks):
                        # 仅对有意义的内容块生成向量
                        if len(chunk.strip()) > 0:
                            embedding = get_embedding(chunk, embedding_client)
                            if not embedding:
                                print(f"    - 跳过无法生成向量的块: chunk {i+1}")
                                continue
                            
                            chunk_id = f"{doc_id_counter}_{i}"
                            es_client.index(
                                index=index_name,
                                id=chunk_id,
                                document={
                                    "file_name": filename,
                                    "content": chunk,
                                    "content_vector": embedding
                                }
                            )
                    doc_id_counter += 1
                else:
                    print(f"  - 跳过空文件: {filename}")

            except Exception as e:
                print(f"  - 处理文件时出错 '{filename}': {e}")
    
    es_client.indices.refresh(index=index_name)

    # 步骤 4: LLM 配置
    llm_cfg = {
        'model': 'qwen-max',
        'model_server': 'dashscope',
        'api_key': os.getenv('DASHSCOPE_API_KEY'),
        'generate_cfg': {
            'top_p': 0.8
        }
    }

    # 步骤 5: RAG 配置 - 激活并配置 Elasticsearch Embedding 后端
    rag_cfg = {
        "rag_backend": "elasticsearch_embedding",  # 关键：指定使用 ES + Embedding 后端
        "es": {
            "host": "http://localhost",
            "port": 9200,
            # "user": "elastic",
            # "password": "euqPcOlHrmW18rtaS-3P",  # 您的 Elasticsearch 密码
            "index_name": index_name, # 使用刚刚创建的向量索引
            "use_embedding": True,  # 启用 embedding 向量搜索
            "embedding_model": "text-embedding-v4",  # 指定 embedding 模型
            "embedding_dimensions": 1024,  # embedding 维度
            "similarity_threshold": 0.5,  # 相似度阈值，低于此值的结果会被过滤
        },
        "parser_page_size": 500  # 文档分块大小
    }

    # 步骤 6: 系统指令
    system_instruction = '''你是一个基于本地知识库的AI助手。
请根据用户的问题，利用检索工具从知识库中查找最相关的信息，并结合这些信息给出专业、准确的回答。
系统会通过语义搜索（向量相似度）找到与问题最相关的文档片段，然后进行推理回答。'''

    # 步骤 7: 获取文件夹下所有文件
    file_dir = os.path.join(os.path.dirname(__file__), 'docs')
    files = []
    if os.path.exists(file_dir):
        for file in os.listdir(file_dir):
            file_path = os.path.join(file_dir, file)
            if os.path.isfile(file_path):
                files.append(file_path)
    print('知识库文件列表:', files)

    # 步骤 8: 创建智能体实例
    print(f'RAG配置中的embedding信息: use_embedding={rag_cfg["es"]["use_embedding"]}, embedding_model={rag_cfg["es"]["embedding_model"]}')
    bot = Assistant(
        llm=llm_cfg,
        system_message=system_instruction,
        files=files,
        rag_cfg=rag_cfg
    )
    return bot


def main():
    """启动 Web 图形界面"""
    try:
        print("正在启动 AI 助手 Web 界面 (Elasticsearch + Embedding 后端)...")
        bot = init_agent_service()
        if bot is None:
            print("初始化助手服务失败，无法启动Web界面。")
            return
        chatbot_config = {
            'prompt.suggestions': [
                '介绍下雇主责任险',
                '雇主责任险和工伤保险有什么主要区别？',
                '介绍一下平安商业综合责任保险（亚马逊）的保障范围。',
                '施工保主要适用于哪些场景？',
            ]
        }
        WebUI(bot, chatbot_config=chatbot_config).run()
    except Exception as e:
        print(f"启动 Web 界面失败: {e}")
        print("请检查网络连接、API Key 以及 Elasticsearch 服务是否正常运行。")


if __name__ == '__main__':
    main()