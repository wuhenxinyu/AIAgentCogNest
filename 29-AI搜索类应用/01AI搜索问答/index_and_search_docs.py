"""
索引并搜索文档,
1. 连接到 Elasticsearch
2. 创建一个索引
3. 索引文档（添加数据）
4. 执行搜索
"""
import os
from elasticsearch import Elasticsearch
from pypdf import PdfReader
import warnings

# 忽略来自 pypdf 的特定用户警告
warnings.filterwarnings("ignore", category=UserWarning, module='pypdf')


def index_and_search_documents():
    """
    连接到ES，索引docs文件夹下的文档，并执行搜索。
    """
    # --- 1. 连接到 Elasticsearch ---
    # 请确保将 "YOUR_ELASTIC_PASSWORD" 替换为您的真实密码
    try:
        # client = Elasticsearch(
        #     "https://localhost:9200"  # 1. 使用 https
            # basic_auth=("elastic", "euqPcOlHrmW18rtaS-3P"),
            # verify_certs=False  # 2. 忽略SSL证书验证 (等同于 curl -k)
        # )
        client = Elasticsearch(
            ['http://localhost:9200'],
            # http_auth=('elastic', 'password'),  # 只有在设置了密码时才需要
            # 重试配置
            max_retries=3,
            retry_on_timeout=True,
            verify_certs=False
        )
        if not client.ping():
            print("无法连接到 Elasticsearch，请检查服务是否正在运行以及密码是否正确。")
            return
        print("成功连接到 Elasticsearch！")
    except Exception as e:
        print(f"连接时发生错误: {e}")
        return

    # --- 2. 创建一个索引 ---
    index_name = "pingan_employer_insurance"
    if client.indices.exists(index=index_name):
        print(f"索引 '{index_name}' 已存在，正在删除旧索引...")
        client.indices.delete(index=index_name)

    print(f"正在创建新索引 '{index_name}'...")
    # 在创建索引时可以定义映射（mapping），这里为了简单起见使用默认映射
    client.indices.create(index=index_name)

    # --- 3. 索引文档（添加数据） ---
    # 使用绝对路径而不是相对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    docs_folder = os.path.join(current_dir, 'docs')
    if not os.path.exists(docs_folder):
        print(f"错误：未找到 '{docs_folder}' 文件夹。")
        return

    print("\n正在从 'docs' 文件夹读取并索引文档...")
    doc_id_counter = 1
    for filename in os.listdir(docs_folder):
        file_path = os.path.join(docs_folder, filename)
        if os.path.isfile(file_path):
            try:
                content = ""
                if filename.lower().endswith('.pdf'):
                    # 使用 pypdf 读取 PDF
                    reader = PdfReader(file_path)
                    for page in reader.pages:
                        # 从当前页面提取文本内容。如果提取的文本为空（即 page.extract_text() 返回 None），
                        # 则使用空字符串 "" 代替，避免出现 None 值。
                        # 最后将提取的文本累加到 content 变量中。
                        content += page.extract_text() or ""
                elif filename.lower().endswith('.txt'):
                    # 读取文本文件
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                else:
                    print(f"  - 跳过不支持的文件类型: {filename}")
                    continue
                
                if content.strip():
                    # 将文档内容索引到 Elasticsearch
                    client.index(
                        index=index_name,
                        id=doc_id_counter,
                        document={
                            "file_name": filename,
                            "content": content
                        }
                    )
                    print(f"  - 成功索引文档: {filename} (ID: {doc_id_counter})")
                    doc_id_counter += 1
                else:
                    print(f"  - 跳过空文件: {filename}")

            except Exception as e:
                print(f"  - 处理文件时出错 '{filename}': {e}")
    
    if doc_id_counter == 1:
        print("\n'docs' 文件夹中没有可以索引的文档。")
        return

    print("\n所有文档索引完成！")
    # 强制刷新索引，确保数据立即可搜
    client.indices.refresh(index=index_name)

    # --- 4. 执行搜索 ---
    search_query = "工伤保险和雇主险有什么区别？"
    print(f"\n正在执行搜索，查询语句: '{search_query}'")

    response = client.search(
        index=index_name,
        query={
            "match": {
                "content": {
                    "query": search_query,
                    "operator": "and" # 使用 and 操作符，要求所有词都出现，更精确
                }
            }
        },
        size=3 # 返回最相关的3个结果
    )

    # --- 5. 显示搜索结果 ---
    print("\n--- 搜索结果 ---")
    hits = response['hits']['hits']
    if not hits:
        print("没有找到匹配的文档。")
    else:
        for i, hit in enumerate(hits):
            print(f"\n--- 结果 {i+1} ---")
            print(f"来源文件: {hit['_source']['file_name']}")
            print(f"相关度得分: {hit['_score']:.2f}")
            # 为了简洁，只显示部分匹配内容片段
            # ES 的高亮功能可以更精确地显示匹配片段，这里作简化处理
            content_preview = hit['_source']['content'].strip().replace('\n', ' ')
            print(f"内容预览: {content_preview[:200]}...")

if __name__ == '__main__':
    index_and_search_documents()