"""
示例：使用FAISS进行向量搜索，同时返回元数据
FAISS 专注于其最擅长的高速向量检索，而元数据的存储和管理可以交给专业的数据库系统负责，实现了架构上的解耦和高效协同。
"""
import os
import numpy as np
import faiss
from openai import OpenAI

# Step1. 初始化 API 客户端
try:
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
except Exception as e:
    print("初始化OpenAI客户端失败，请检查环境变量'DASHSCOPE_API_KEY'是否已设置。")
    print(f"错误信息: {e}")
    exit()

# Step2. 准备示例文本和元数据
# 在实际应用中，这些数据可能来自数据库、文件等
documents = [
    {
        "id": "doc1",
        "text": "迪士尼乐园的门票一经售出，原则上不予退换。但在特殊情况下，如恶劣天气导致园区关闭，可在官方指引下进行改期或退款。",
        "metadata": {"source": "official_faq_v1.pdf", "category": "退票政策", "author": "Admin"}
    },
    {
        "id": "doc2",
        "text": "购买“奇妙年卡”的用户，可以享受一年内多次入园的特权，并且在餐饮和购物时有折扣。",
        "metadata": {"source": "annual_pass_rules.docx", "category": "会员权益", "author": "MarketingDept"}
    },
    {
        "id": "doc3",
        "text": "对于在线购买的迪士尼门票，如果需要退票，必须在票面日期前48小时通过原购买渠道提交申请，并可能收取手续费。",
        "metadata": {"source": "online_policy.html", "category": "退票政策", "author": "E-commerceTeam"}
    },
    {
        "id": "doc4",
        "text": "园区内的“加勒比海盗”项目因年度维护，将于下周暂停开放。",
        "metadata": {"source": "maintenance_notice.txt", "category": "园区公告", "author": "OpsDept"}
    }
]

# Step3. 创建元数据存储和向量列表
# 我们使用一个简单的列表来存储元数据。列表的索引将作为FAISS的ID。
# 这种方式简单直接，适用于中小型数据集。
# 对于大型数据集，可以考虑使用字典或数据库（如Redis, SQLite）
metadata_store = []
vectors_list = []
vector_ids = []

print("正在为文档生成向量...")
for i, doc in enumerate(documents):
    try:
        # 调用API生成向量
        completion = client.embeddings.create(
            model="text-embedding-v4",
            input=doc["text"],
            dimensions=1024,
            encoding_format="float"
        )
        
        # 获取向量
        vector = completion.data[0].embedding
        vectors_list.append(vector)
        
        # 存储元数据，并使用列表索引作为唯一ID
        metadata_store.append(doc)
        vector_ids.append(i) # 自定义ID与列表索引一致
        
        print(f"  - 已处理文档 {i+1}/{len(documents)}")

    except Exception as e:
        print(f"处理文档 '{doc['id']}' 时出错: {e}")
        continue

# 将向量列表转换为NumPy数组，FAISS需要这种格式
vectors_np = np.array(vectors_list).astype('float32')
vector_ids_np = np.array(vector_ids)

# Step4. 构建并填充 FAISS 索引
dimension = 1024  # 向量维度
k = 3             # 查找最近的3个邻居

# 创建一个基础的L2距离索引
index_flat_l2 = faiss.IndexFlatL2(dimension)

# 使用IndexIDMap来包装基础索引，能够映射我们自定义的ID
# 这就是关联向量和元数据的关键！
index = faiss.IndexIDMap(index_flat_l2)

# 将向量和它们对应的ID添加到索引中
index.add_with_ids(vectors_np, vector_ids_np)

print(f"\nFAISS 索引已成功创建，共包含 {index.ntotal} 个向量。")


# Step5. 执行搜索并检索元数据
query_text = "我想了解一下迪士尼门票的退款流程"
print(f"\n正在为查询文本生成向量: '{query_text}'")

try:
    # 为查询文本生成向量
    query_completion = client.embeddings.create(
        model="text-embedding-v4",
        input=query_text,
        dimensions=1024,
        encoding_format="float"
    )
    query_vector = np.array([query_completion.data[0].embedding]).astype('float32')

    # 在FAISS索引中执行搜索
    # search方法返回两个NumPy数组：
    # D: 距离 (distances)
    # I: 索引/ID (indices/IDs)
    distances, retrieved_ids = index.search(query_vector, k)

    # Step6. 展示结果
    print("\n--- 搜索结果 ---")
    # `retrieved_ids[0]` 包含与查询最相似的k个向量的ID
    for i in range(k):
        doc_id = retrieved_ids[0][i]
        
        # 检查ID是否有效
        if doc_id == -1:
            print(f"\n排名 {i+1}: 未找到更多结果。")
            continue

        # 使用ID从我们的元数据存储中检索信息
        retrieved_doc = metadata_store[doc_id]
        
        print(f"\n--- 排名 {i+1} (相似度得分/距离: {distances[0][i]:.4f}) ---")
        print(f"ID: {doc_id}")
        print(f"原始文本: {retrieved_doc['text']}")
        print(f"元数据: {retrieved_doc['metadata']}")

except Exception as e:
    print(f"执行搜索时发生错误: {e}")

