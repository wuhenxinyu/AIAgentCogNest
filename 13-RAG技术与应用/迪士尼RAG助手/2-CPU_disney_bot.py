"""
迪士尼客服 RAG 助手 - 完整 Python 代码示例

本代码实现了《RAG（Retrieval Augmented Generation）技术与应用》案例中描述的全流程。
它将演示如何处理多种格式的文档（Word），提取文本、表格和图片，
然后使用 Embedding 模型（text-embedding-v3, CLIP）和 FAISS 向量库构建一个
能够回答文本和图片相关问题的智能问答系统。

在运行前，请确保完成以下准备工作：

1. 安装所有必需的 Python 库:
   pip install openai "faiss-cpu" python-docx PyMuPDF Pillow pytesseract transformers torch requests

2. 安装 Google Tesseract OCR 引擎:
   - Windows: 从 https://github.com/UB-Mannheim/tesseract/wiki 下载并安装。
   - macOS: brew install tesseract
   - Linux (Ubuntu): sudo apt-get install tesseract-ocr
   请确保 tesseract 的可执行文件路径已添加到系统的 PATH 环境变量中。

3. 设置环境变量:
   - DASHSCOPE_API_KEY: 您从阿里云百炼平台获取的 API Key。
   - HF_TOKEN: (可选) 您的 Hugging Face Token，用于下载 CLIP 模型，避免手动确认。

"""
import os
import re
import numpy as np
import faiss
from openai import OpenAI
from docx import Document as DocxDocument
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from transformers import CLIPProcessor, CLIPModel
import torch

# Step0. 全局配置与模型加载
os.environ["DASHSCOPE_API_KEY"] = "sk-siRF78nIxVVBKekhvZF6POAzSrFXymXwzCFj4YT6SzFIlvWA"
# 检查环境变量
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("错误：请设置 'DASHSCOPE_API_KEY' 环境变量。")

# 初始化百炼兼容的 OpenAI 客户端
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://api.fe8.cn/v1"
)

# 加载 CLIP 模型用于图像处理 (如果本地没有会自动下载)
print("正在加载 CLIP 模型...")
try:
    # 从本地加载模型
    clip_model = CLIPModel.from_pretrained("/dev/shm/models/openai/clip-vit-base-patch32", local_files_only=True)
    # 从本地加载CLIP预处理器,用于处理输入图像和文本
    # CLIPProcessor包含了图像预处理和文本分词的功能
    # local_files_only=True表示只从本地加载,不从网络下载
    clip_processor = CLIPProcessor.from_pretrained("/dev/shm/models/openai/clip-vit-base-patch32", local_files_only=True)
    print("CLIP 模型从本地加载成功。")
except Exception as e:
    # 如果本地没有模型，尝试从 Hugging Face 下载并保存
    try:
        print("本地模型不存在，正在从 Hugging Face 下载...")
        os.makedirs("./models", exist_ok=True)
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # 保存到本地
        clip_model.save_pretrained("./models/clip-vit-base-patch32")
        clip_processor.save_pretrained("./models/clip-vit-base-patch32")
        print("CLIP 模型下载并保存到本地成功。")
    except Exception as download_error:
        print(f"下载或保存 CLIP 模型失败。错误: {download_error}")
        exit()

# 定义全局变量
DOCS_DIR = "disney_knowledge_base"
PDF_DIR = os.path.join(DOCS_DIR, "pdfs")
IMG_DIR = os.path.join(DOCS_DIR, "images")
TEXT_EMBEDDING_MODEL = "text-embedding-v3"
TEXT_EMBEDDING_DIM = 1024
IMAGE_EMBEDDING_DIM = 512 # CLIP 'vit-base-patch32' 模型的输出维度

# Step1. 文档解析与内容提取
def parse_docx(file_path):
    """解析 DOCX 文件，提取文本和表格（转为Markdown）。"""
    doc = DocxDocument(file_path)
    content_chunks = []
    
    # 遍历文档中的每个元素
    for element in doc.element.body:
        # 如果元素是段落（以'p'结尾的标签）
        if element.tag.endswith('p'):
            # 处理段落
            paragraph_text = ""
            # 查找所有文本运行节点并拼接文本
            for run in element.findall('.//w:t', {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}):
                paragraph_text += run.text if run.text else ""
            
            # 如果段落文本不为空，添加到内容块列表
            if paragraph_text.strip():
                content_chunks.append({"type": "text", "content": paragraph_text.strip()})
                
        # 如果元素是表格（以'tbl'结尾的标签）
        elif element.tag.endswith('tbl'):
            # 处理表格，转换为Markdown格式
            md_table = []
            # 获取对应的表格对象
            table = [t for t in doc.tables if t._element is element][0]
            
            if table.rows:
                # 提取并格式化表头
                header = [cell.text.strip() for cell in table.rows[0].cells]
                md_table.append("| " + " | ".join(header) + " |")
                # 添加Markdown表格分隔行
                md_table.append("|" + "---|"*len(header))
                
                # 处理数据行
                for row in table.rows[1:]:
                    row_data = [cell.text.strip() for cell in row.cells]
                    md_table.append("| " + " | ".join(row_data) + " |")
                
                # 将表格内容合并为字符串
                table_content = "\n".join(md_table)
                # 如果表格内容不为空，添加到内容块列表
                if table_content.strip():
                    content_chunks.append({"type": "table", "content": table_content})
    
    # 返回所有提取的内容块
    return content_chunks

def parse_pdf(file_path, image_dir):
    """解析 PDF 文件，提取文本和图片。parse_pdf 使用 fitz (PyMuPDF) 库打开并逐页读取 PDF 文档。它的核心目标是将 PDF 这种复合式文档，拆解成纯文
本和独立的图片文件，以便 RAG 模型后续能分别处理和理解。"""
    doc = fitz.open(file_path)
    content_chunks = []
    for page_num, page in enumerate(doc):
        # 提取文本
        # 代码遍历每一页，使用 get_text() 方法抓取该页所有的纯文本内容。
        # 代码将每页的文本保存为一个独立的区块 (chunk)，并附上页码。    
        text = page.get_text("text")
        content_chunks.append({"type": "text", "content": text, "page": page_num + 1})
        # 提取图片
        # 侦测并提取页面中的所有嵌入图片。
        # 代码将每张图片的二进制数据读取出来，以唯一的文件名（包含原始文件名、页码和图片索引）保存到指定的
        # image_dir 目录下。同时，它会记录图片的存储路径。
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            # 提取图片扩展名
            image_ext = base_image["ext"]
            # 构建图片保存路径
            image_path = os.path.join(image_dir, f"{os.path.basename(file_path)}_p{page_num+1}_{img_index}.{image_ext}")
            # 保存图片到指定目录
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            # 将图片路径和页码信息添加到内容块列表
            content_chunks.append({"type": "image", "path": image_path, "page": page_num + 1})
    # 返回所有提取的内容块
    return content_chunks

# 定义一个函数 image_to_text，用于将图片中的文字提取出来
# 函数的作用是“看图识字”。
# 它接收一个图片文件的路径，通过OCR，提取图片中包含的文字信息，适合扫描版 PDF 或截图。
def image_to_text(image_path):
    """对图片进行OCR和CLIP描述。"""
    try:
        image = Image.open(image_path)
        # OCR
        # 使用pytesseract进行OCR文字识别,支持中文和英文双语识别,去除首尾空格
        ocr_text = pytesseract.image_to_string(image, lang='chi_sim+eng').strip()
        return {"ocr": ocr_text}
    except Exception as e:
        print(f"处理图片失败 {image_path}: {e}")
        return {"ocr": ""}

# Step2. Embedding 与索引构建
def get_text_embedding(text):
    if not text or not isinstance(text, str):
        raise ValueError("输入文本不能为空且必须是字符串！")
    """获取文本的 Embedding。"""
    response = client.embeddings.create(
        model=TEXT_EMBEDDING_MODEL,
        input=text,
        dimensions=TEXT_EMBEDDING_DIM # 文本嵌入维度
    )
    return response.data[0].embedding

def get_image_embedding(image_path):
    """获取图片的 Embedding。"""
    # 1. 加载图像
    image = Image.open(image_path)
    # 2. 使用CLIP预处理器处理输入图像（只返回pixel_values）
    # images=image: 输入的图像数据
    # return_tensors="pt": 返回PyTorch格式的张量
    inputs = clip_processor(images=image, return_tensors="pt")
    

    # 5. 推理设置：设置PyTorch只使用单线程进行计算,避免多线程导致的性能开销
    # 因为这里只是简单的前向推理,不需要并行计算
    torch.set_num_threads(1)
    
    # 使用torch.no_grad()上下文管理器来禁用梯度计算
    # 因为在推理阶段不需要计算梯度和进行反向传播
    # 这样可以减少内存使用并提高计算速度
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)

    # 将CLIP模型输出的图像特征向量转换为numpy数组格式并返回第一个样本
    # image_features的shape为[batch_size, feature_dim]，这里我们只取第一个样本
    # numpy()方法将PyTorch张量转换为NumPy数组,便于后续处理
    # [0]取第一个样本,因为batch_size=1
    return image_features.numpy() 

def get_clip_text_embedding(text):
    """使用CLIP的文本编码器获取文本的Embedding。"""
    inputs = clip_processor(text=text, return_tensors="pt")
    
    # 使用CLIP模型的文本编码器将输入文本转换为特征向量
    # 使用torch.no_grad()上下文管理器禁用梯度计算,因为这里只需要前向推理,不需要反向传播
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)

    return text_features.numpy() 

def build_knowledge_base(docs_dir, img_dir, pdf_dir):

    """构建完整的知识库，包括解析、切片、Embedding和索引。"""
    print("\n--- 步骤 1 & 2: 正在解析、Embedding并索引知识库 ---")
    
    # 存储所有文档和图片的元数据信息
    metadata_store = []
    # 存储所有文本内容的向量表示
    text_vectors = []
    # 存储所有图片的向量表示
    image_vectors = []
    
    doc_id_counter = 0

    # 处理Word文档
    # 遍历docs_dir目录下的所有文件
    for filename in os.listdir(docs_dir):
        # 跳过隐藏文件(以.开头)和目录
        if filename.startswith('.') or os.path.isdir(os.path.join(docs_dir, filename)):
            continue
            
        # 构建完整的文件路径
        file_path = os.path.join(docs_dir, filename)
        if filename.endswith(".docx"):
            print(f"  - 正在处理: {filename}")
            chunks = parse_docx(file_path)
            
            # 遍历每个文档块(文本或表格)
            for chunk in chunks:
                # 创建基础元数据字典
                metadata = {
                    "id": doc_id_counter,  # 文档块的唯一标识符
                    "source": filename,     # 来源文件名
                    "page": 1              # 页码(Word文档默认为1)
                }
                
                # 处理文本或表格类型的块
                if chunk["type"] in ["text", "table"]:
                    # 获取内容文本
                    text = chunk["content"]
                    # 跳过空文本
                    if not text.strip():
                        continue
                    
                    # 更新元数据
                    metadata["type"] = "text"  # 将类型统一标记为text
                    metadata["content"] = text  # 存储实际内容
                    
                    # 获取文本的向量表示
                    vector = get_text_embedding(text)
                    # 将向量和元数据分别存储到对应列表
                    text_vectors.append(vector)
                    metadata_store.append(metadata)
                    # 递增文档ID计数器
                    doc_id_counter += 1

    # 处理PDF文件
    print("  - 正在处理PDF文件...")
    for pdf_filename in os.listdir(pdf_dir):
        if pdf_filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, pdf_filename)
            print(f"    - 处理PDF: {pdf_filename}")
            # 解析PDF文件
            pdf_chunks = parse_pdf(pdf_path,img_dir)
            # 遍历每个PDF文档块
            for chunk in pdf_chunks:
                # 创建基础元数据字典
                metadata = {
                    "id": doc_id_counter,  # 文档块的唯一标识符
                    "source": pdf_filename,     # 来源文件名
                    "page": chunk["page"]             # 页码(Word文档默认为1)
                }
                # 处理文本类型的块和图片
                if chunk["type"] == "text":
                    # 获取内容文本
                    text = chunk["content"]
                    # 跳过空文本
                    if not text.strip():
                        continue
                     # 更新元数据
                    metadata["type"] = "text"  # 将类型统一标记为text
                    metadata["content"] = text  # 存储实际内容
                    # 获取文本的向量表示
                    vector = get_text_embedding(text)
                    # 将向量和元数据分别存储到对应列表  
                    text_vectors.append(vector)
                    metadata_store.append(metadata)
                    # 递增文档ID计数器
                    doc_id_counter += 1
                elif chunk["type"] == "image":
                    # 获取图片路径
                    img_path = chunk["path"]
                    # 提取图片中的文字
                    img_text_info = image_to_text(img_path)
                    # 获取图片向量表示
                    vector = get_image_embedding(img_path)
                    # 更新元数据
                    metadata["type"] = "image"
                    metadata["path"] = img_path
                    metadata["ocr"] = img_text_info["ocr"]
                    # 存储图片向量和元数据
                    image_vectors.append(vector)
                    metadata_store.append(metadata)
                    # 递增文档ID计数器
                    doc_id_counter += 1

    # 处理images目录中的独立图片文件
    print("  - 正在处理独立图片文件...")
    for img_filename in os.listdir(img_dir):
        if img_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            img_path = os.path.join(img_dir, img_filename)
            print(f"    - 处理图片: {img_filename}")
            
            img_text_info = image_to_text(img_path)
            
            metadata = {
                "id": doc_id_counter,
                "source": f"独立图片: {img_filename}",
                "type": "image",
                "path": img_path,
                "ocr": img_text_info["ocr"],
                "page": 1
            }
            
            vector = get_image_embedding(img_path)
            image_vectors.append(vector)
            metadata_store.append(metadata)
            doc_id_counter += 1

    # 创建 FAISS 索引
    # 文本索引
    # 创建一个基于L2距离(欧氏距离)的FAISS索引,用于存储文本向量
    # TEXT_EMBEDDING_DIM指定向量维度为1024
    text_index = faiss.IndexFlatL2(TEXT_EMBEDDING_DIM)
    
    # 创建一个支持ID映射的索引包装器
    # 这样可以将向量与文档ID关联起来,方便后续检索时找到对应文档
    text_index_map = faiss.IndexIDMap(text_index)
    
    # 从metadata_store中提取所有文本类型文档的ID
    # 使用列表推导式,只选择type为text的文档的id
    text_ids = [m["id"] for m in metadata_store if m["type"] == "text"]
    
    # 只有当存在文本向量时才添加到索引中
    if text_vectors:
        # 将文本向量和对应的ID添加到索引中
        # np.array将列表转换为numpy数组
        # astype('float32')确保向量数据类型为32位浮点数
        # add_with_ids方法同时添加向量和对应的ID
        text_index_map.add_with_ids(np.array(text_vectors).astype('float32'), np.array(text_ids))
    
    # 图像索引
    image_index = faiss.IndexFlatL2(IMAGE_EMBEDDING_DIM)
    image_index_map = faiss.IndexIDMap(image_index)
    image_ids = [m["id"] for m in metadata_store if m["type"] == "image"]
    if image_vectors:  # 只有当有图像向量时才添加到索引
        image_index_map.add_with_ids(np.array(image_vectors).astype('float32'), np.array(image_ids))
    
    print(f"索引构建完成。共索引 {len(text_vectors)} 个文本片段和 {len(image_vectors)} 张图片。")
    
    return metadata_store, text_index_map, image_index_map

# Step3. RAG 问答流程
def rag_ask(query, metadata_store, text_index, image_index, k=3):
    """
    执行完整的 RAG 流程：检索 -> 构建Prompt -> 生成答案
    """
    print(f"\n--- 收到用户提问: '{query}' ---")
    
    # 步骤 1: 检索
    print("  - 步骤 1: 向量化查询并进行检索...")
    retrieved_context = []
    
    # 文本检索
    # 将用户查询转换为向量表示
    # 1. 使用get_text_embedding获取查询文本的embedding向量
    # 2. 将向量转换为numpy数组并指定数据类型为float32
    # 3. 添加一个维度使其成为2D数组,因为faiss.search期望的输入是批量查询
    query_text_vec = np.array([get_text_embedding(query)]).astype('float32')
    
    # 在文本向量索引中搜索最相似的k个文档
    # distances: 返回每个查询结果与查询向量的L2距离
    # text_ids: 返回检索到的文档ID
    distances, text_ids = text_index.search(query_text_vec, k)
    for i, doc_id in enumerate(text_ids[0]):
        if doc_id != -1:
            # 通过ID在元数据中查找
            match = next((item for item in metadata_store if item["id"] == doc_id), None)
            if match:
                retrieved_context.append(match)
                print(f"    - 文本检索命中 (ID: {doc_id}, 距离: {distances[0][i]:.4f})")

    # 图像检索 (使用CLIP文本编码器)
    # 简单判断是否需要检索图片
    if any(keyword in query.lower() for keyword in ["海报", "图片", "长什么样", "看看", "万圣节", "聚在一起"]):
        print("  - 检测到图像查询关键词，执行图像检索...")
        query_clip_vec = np.array([get_clip_text_embedding(query)]).astype('float32')
        distances, image_ids = image_index.search(query_clip_vec, 1) # 只找最相关的1张图
        for i, doc_id in enumerate(image_ids[0]):
            if doc_id != -1:
                match = next((item for item in metadata_store if item["id"] == doc_id), None)
                if match:
                    # 将OCR内容也加入上下文
                    context_text = f"找到一张相关图片，图片路径: {match['path']}。图片上的文字是: '{match['ocr']}'"
                    retrieved_context.append({"type": "image_context", "content": context_text, "metadata": match})
                    print(f"    - 图像检索命中 (ID: {doc_id}, 距离: {distances[0][i]:.4f})")
    
    # 步骤 2: 构建 Prompt 并生成答案
    print("  - 步骤 2: 构建 Prompt...")
    context_str = ""
    for i, item in enumerate(retrieved_context):
        content = item.get('content', '')
        source = item.get('metadata', {}).get('source', item.get('source', '未知来源'))
        context_str += f"背景知识 {i+1} (来源: {source}):\n{content}\n\n"
        
    prompt = f"""你是一个迪士尼客服助手。请根据以下背景知识，用友好和专业的语气回答用户的问题。请只使用背景知识中的信息，不要自行发挥。

[背景知识]
{context_str}
[用户问题]
{query}
"""
    print("--- Prompt Start ---")
    print(prompt)
    print("--- Prompt End ---")
    
    print("\n  - 步骤 3: 调用 LLM 生成最终答案...")
    try:
        completion = client.chat.completions.create(
            model="qwen-plus", # 使用一个强大的模型进行生成
            messages=[
                {"role": "system", "content": "你是一个迪士尼客服助手。"},
                {"role": "user", "content": prompt}
            ]
        )
        final_answer = completion.choices[0].message.content
        
        # 答案后处理：如果上下文中包含图片，提示用户
        image_path_found = None
        for item in retrieved_context:
            if item.get("type") == "image_context":
                image_path_found = item.get("metadata", {}).get("path")
                break
        
        if image_path_found:
            final_answer += f"\n\n(同时，我为您找到了相关图片，路径为: {image_path_found})"

    except Exception as e:
        final_answer = f"调用LLM时出错: {e}"

    print("\n--- 最终答案 ---")
    print(final_answer)
    return final_answer

# --- 主函数 ---
if __name__ == "__main__":
    # 1. 构建知识库 (一次性离线过程)
    metadata_store, text_index, image_index = build_knowledge_base(DOCS_DIR, IMG_DIR,PDF_DIR)
    
    # 2. 开始问答
    print("\n=============================================")
    print("迪士尼客服RAG助手已准备就绪，开始模拟提问。")
    print("=============================================")
    
    # 案例1: 文本问答
    rag_ask(
        query="我想了解一下迪士尼门票的退款流程",
        metadata_store=metadata_store,
        text_index=text_index,
        image_index=image_index
    )
    
    print("\n---------------------------------------------\n")
    
    # 案例2: 多模态问答
    rag_ask(
        query="最近万圣节的活动海报是什么",
        metadata_store=metadata_store,
        text_index=text_index,
        image_index=image_index
    )
    
    print("\n---------------------------------------------\n")
    
    # 案例3: 年卡相关问答
    rag_ask(
        query="迪士尼年卡有什么优惠",
        metadata_store=metadata_store,
        text_index=text_index,
        image_index=image_index
    )

    print("\n---------------------------------------------\n")
    
    # 案例4: 投诉相关问答
    rag_ask(
        query="客户经理被投诉了，投诉一次扣多少分",
        metadata_store=metadata_store,
        text_index=text_index,
        image_index=image_index
    )