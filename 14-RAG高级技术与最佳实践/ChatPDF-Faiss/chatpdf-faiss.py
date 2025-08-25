"""
DeepSeek + Faiss搭建本地知识库检索ChatPDF-Faiss
"""
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks.manager import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Tuple
import os
import pickle
os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here'
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

def extract_text_with_page_numbers(pdf) -> Tuple[str, List[Tuple[str, int]]]:
    """
    从PDF中提取文本并记录每个字符对应的页码;
    使用PyPDF2库的PdfReader从PDF文件中提取文本在提取过程中记录每行文本对应的页码，便于后续溯源;
    
    
    参数:
        pdf: PDF文件对象
    
    返回:
        text: 提取的文本内容
        char_page_mapping: 每个字符对应的页码列表
    """
    text = ""
    char_page_mapping = []

    # 遍历PDF中的每一页,页码从1开始
    for page_number, page in enumerate(pdf.pages, start=1):
        # 提取当前页面的文本内容
        extracted_text = page.extract_text()
        # 如果成功提取到文本
        if extracted_text:
            # 将提取的文本添加到总文本中
            text += extracted_text
            # 为当前页面的每个字符记录页码
            char_page_mapping.extend([page_number] * len(extracted_text))
        # 如果当前页面没有文本内容
        else:
            # 打印提示信息
            print(f"No text found on page {page_number}.")

    return text, char_page_mapping

def process_text_with_splitter(text: str, char_page_mapping: List[int], save_path: str = None) -> FAISS:
    """
    处理文本并创建向量存储;使用RecursiveCharacterTextSplitter将长文本分割成小块，便于向量化处理;
    每个文本块的页码信息会被记录下来，后续查询时可以根据页码信息进行溯源。
    
    参数:
        text: 提取的文本内容
        char_page_mapping: 每个字符对应的页码列表
        save_path: 可选，保存向量数据库的路径
    
    返回:
        knowledgeBase: 基于FAISS的向量存储对象
    """
    # 创建文本分割器，用于将长文本分割成小块
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    # 分割文本
    chunks = text_splitter.split_text(text)
    print(f"文本被分割成 {len(chunks)} 个块。")
        

    # 创建DashScope嵌入模型实例
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=DASHSCOPE_API_KEY
    )
    
    # 从文本块创建知识库
    # 使用FAISS.from_texts方法创建向量数据库
    # 参数chunks: 文本块列表，每个文本块将被转换为向量
    # 参数embeddings: 用于将文本转换为向量的嵌入模型
    # 返回一个FAISS向量数据库实例，用于后续的相似度搜索
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    print("已从文本块创建知识库。")
    
    # 为每个文本块找到对应的页码信息
    page_info = {}
    current_pos = 0
    
    for chunk in chunks:
        chunk_start = current_pos
        chunk_end = current_pos + len(chunk)
        
        # 找到这个文本块中字符对应的页码
        chunk_pages = char_page_mapping[chunk_start:chunk_end]
        
        # 取页码的众数（出现最多的页码）作为该块的页码
        if chunk_pages:
            # 统计每个页码出现的次数
            page_counts = {}
            for page in chunk_pages:
                page_counts[page] = page_counts.get(page, 0) + 1
            
            # 找到出现次数最多的页码
            most_common_page = max(page_counts, key=page_counts.get)
            page_info[chunk] = most_common_page
        else:
            page_info[chunk] = 1  # 默认页码
        
        current_pos = chunk_end
    
    knowledgeBase.page_info = page_info
    print(f'页码映射完成，共 {len(page_info)} 个文本块')
    
    # 如果提供了保存路径，则保存向量数据库和页码信息
    if save_path:
        # 确保目录存在
        os.makedirs(save_path, exist_ok=True)
        
        # 保存FAISS向量数据库
        knowledgeBase.save_local(save_path)
        print(f"向量数据库已保存到: {save_path}")
        
        # 保存页码信息到同一目录
        with open(os.path.join(save_path, "page_info.pkl"), "wb") as f:
            pickle.dump(page_info, f)
        print(f"页码信息已保存到: {os.path.join(save_path, 'page_info.pkl')}")
    
    return knowledgeBase

def load_knowledge_base(load_path: str, embeddings = None) -> FAISS:
    """
    从磁盘加载向量数据库和页码信息
    
    参数:
        load_path: 向量数据库的保存路径
        embeddings: 可选，嵌入模型。如果为None，将创建一个新的DashScopeEmbeddings实例
    
    返回:
        knowledgeBase: 加载的FAISS向量数据库对象
    """
    # 如果没有提供嵌入模型，则创建一个新的
    if embeddings is None:
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v1",
            dashscope_api_key=DASHSCOPE_API_KEY,
        )
    
    # 加载FAISS向量数据库，添加allow_dangerous_deserialization=True参数以允许反序列化
    knowledgeBase = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
    print(f"向量数据库已从 {load_path} 加载。")
    
    # 加载页码信息
    page_info_path = os.path.join(load_path, "page_info.pkl")
    if os.path.exists(page_info_path):
        with open(page_info_path, "rb") as f:
            page_info = pickle.load(f)
        knowledgeBase.page_info = page_info
        print("页码信息已加载。")
    else:
        print("警告: 未找到页码信息文件。")
    
    return knowledgeBase

# 读取PDF文件
pdf_reader = PdfReader('./浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf')
# 提取文本和页码信息
text, char_page_mapping = extract_text_with_page_numbers(pdf_reader)


print(f"提取的文本长度: {len(text)} 个字符。")
    
# 处理文本并创建知识库，同时保存到磁盘
save_dir = "./vector_db"
knowledgeBase = process_text_with_splitter(text, char_page_mapping, save_path=save_dir)

# 示例：如何加载已保存的向量数据库
# 注释掉以下代码以避免在当前运行中重复加载
"""
# 创建嵌入模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY,
)
# 从磁盘加载向量数据库
loaded_knowledgeBase = load_knowledge_base("./vector_db", embeddings)
# 使用加载的知识库进行查询
docs = loaded_knowledgeBase.similarity_search("客户经理每年评聘申报时间是怎样的？")

# 直接使用FAISS.load_local方法加载（替代方法）
# loaded_knowledgeBase = FAISS.load_local("./vector_db", embeddings, allow_dangerous_deserialization=True)
# 注意：使用这种方法加载时，需要手动加载页码信息
"""


from langchain_community.llms import Tongyi
llm = Tongyi(model_name="deepseek-v3", dashscope_api_key=DASHSCOPE_API_KEY) # qwen-turbo

# 设置查询问题
query = "客户经理被投诉了，投诉一次扣多少分"
query = "客户经理每年评聘申报时间是怎样的？"
if query:
    # 执行相似度搜索，找到与查询相关的文档
    docs = knowledgeBase.similarity_search(query,k=10)

    # 加载问答链
    chain = load_qa_chain(llm, chain_type="stuff")

    # 准备输入数据
    input_data = {"input_documents": docs, "question": query}

    # TODO 新增，使用回调函数跟踪API调用成本
    with get_openai_callback() as cost:
        # 执行问答链
        response = chain.invoke(input=input_data)
        print(f"查询已处理。成本: {cost}")
        print(response["output_text"])
        print("来源:")

    # 记录唯一的页码
    unique_pages = set()

    # 显示每个文档块的来源页码
    for doc in docs:
        text_content = getattr(doc, "page_content", "")
        source_page = knowledgeBase.page_info.get(
            text_content.strip(), "未知"
        )

        if source_page not in unique_pages:
            unique_pages.add(source_page)
            print(f"文本块页码: {source_page}")

