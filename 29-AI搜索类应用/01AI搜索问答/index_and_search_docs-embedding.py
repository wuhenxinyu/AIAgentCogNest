"""
基于 embedding向量的召回:使用 Dashscope 的 text-embedding-v4 模型为文本生成向量，并执行向量搜索。
引入 chunk_text 函数：增加了文本分块函数，它能将任意长度的文本，切分成长度为 4000 字符、并有 200 字符重叠的文本块列表。
"""
# query: 工伤保险和雇主险有什么区别？
import os
import warnings
from elasticsearch import Elasticsearch
from pypdf import PdfReader
from openai import OpenAI
import warnings
os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here'
warnings.filterwarnings("ignore")

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


def index_and_search_documents_with_embedding():
    """
    连接到ES，使用 embedding 对文档进行索引，并执行向量搜索。
    """
    # --- 1. 初始化客户端 ---
    # a. Elasticsearch 客户端
    try:
        es_client = Elasticsearch(
            "http://localhost:9200",
            # basic_auth=("elastic", "euqPcOlHrmW18rtaS-3P"),
            verify_certs=False
        )
        if not es_client.ping():
            print("无法连接到 Elasticsearch。")
            return
        print("成功连接到 Elasticsearch！")
    except Exception as e:
        print(f"ES 连接时发生错误: {e}")
        return

    # b. Dashscope (OpenAI-compatible) 客户端
    try:
        embedding_client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    except Exception as e:
        print(f"初始化 embedding 客户端时发生错误: {e}")
        return

    # --- 2. 创建一个支持向量搜索的索引 ---
    index_name = "pingan_employer_insurance_embedding"
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

    # --- 3. 索引文档 ---
    docs_folder = 'docs'
    if not os.path.exists(docs_folder):
        print(f"错误：未找到 '{docs_folder}' 文件夹。")
        return

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
                    # 将长文本分块
                    text_chunks = chunk_text(content)
                    print(f"  - 文件 '{filename}'被切分成 {len(text_chunks)} 个块。")
                    
                    for i, chunk in enumerate(text_chunks):
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

    # --- 4. 执行向量搜索 (k-NN) ---
    search_query = "工伤保险和雇主险有什么区别？"
    print(f"\n正在执行向量搜索，查询语句: '{search_query}'")
    
    # a. 为查询语句生成向量
    query_vector = get_embedding(search_query, embedding_client)

    if not query_vector:
        print("无法为查询生成向量，搜索终止。")
        return
        
    # b. 执行 k-NN 搜索
    response = es_client.search(
        index=index_name,
        knn={
            "field": "content_vector",
            "query_vector": query_vector,
            "k": 3,
            "num_candidates": 100
        },
        size=3
    )

    # --- 5. 显示搜索结果 ---
    print("\n--- 向量搜索结果 ---")
    hits = response['hits']['hits']
    if not hits:
        print("没有找到语义上匹配的文档。")
    else:
        for i, hit in enumerate(hits):
            print(f"\n--- 结果 {i+1} ---")
            print(f"来源文件: {hit['_source']['file_name']}")
            print(f"相关度得分 (cosine similarity): {hit['_score']:.4f}")
            content_preview = hit['_source']['content'].strip().replace('\n', ' ')
            print(f"内容预览: {content_preview[:200]}...")

if __name__ == '__main__':
    index_and_search_documents_with_embedding() 