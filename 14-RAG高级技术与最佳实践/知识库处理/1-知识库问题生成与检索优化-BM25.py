"""
当用户提问与知识切片的相似度不高时，能否通过AI为每个知识切片生成可能的问题，通过问题与问题的匹配来提高检索准确度？
    TO DO：对知识库进行问题生成，并进行检索优化
    • 自动生成多样化问题：为知识库中的每个知识切片自动生成多种类型、不同难度的问题
    generate_questions_for_chunk() : 为单个知识切片生成基础问题-----（doc2query）
    generate_diverse_questions() : 生成更多样化的问题（8个）------（doc2query）
    • 构建双重检索索引：同时构建基于原文内容和生成问题的BM25检索索引
    • 检索评估：针对两种检索方式，进行详细的评估
"""
# 知识库问题生成与检索优化 - BM25版本
# 导入依赖库
import os
import json
import numpy as np
from openai import OpenAI
import pandas as pd
from datetime import datetime
from rank_bm25 import BM25Okapi
import jieba
import re

# 从环境变量中获取 API Key
os.environ['DASHSCOPE_API_KEY'] = 'sk-58f051ae745e4bb19fdca31735105b11'
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')

# 初始化百炼兼容的 OpenAI 客户端
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 预处理AI响应中的JSON格式
def preprocess_json_response(response):
    """预处理AI响应，移除markdown代码块格式"""
    if not response:
        return ""
    
    # 移除markdown代码块格式
    if response.startswith('```json'):
        response = response[7:]  # 移除 ```json
    elif response.startswith('```'):
        response = response[3:]  # 移除 ```
    
    if response.endswith('```'):
        response = response[:-3]  # 移除结尾的 ```
    
    return response.strip()  # 移除首尾空白

# 基于 prompt 生成文本
def get_completion(prompt, model="qwen-turbo-latest"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7, # 控制输出的随机性，0.7是一个常用的默认值

    )
    return response.choices[0].message.content

# 文本预处理和分词
def preprocess_text(text):
    """文本预处理和分词"""
    if not text:
        return []
    
    # 移除标点符号和特殊字符
    text = re.sub(r'[^\w\s]', '', text)
    
    # 使用jieba分词
    words = jieba.lcut(text)
    
    # 过滤停用词和短词
    stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
    # 使用列表推导式过滤分词结果:
    # 1. 保留长度大于1的词(过滤掉单字词)
    # 2. 过滤掉停用词
    # len(word) > 1: 确保词长度大于1个字符
    # word not in stop_words: 确保词不在停用词列表中
    words = [word for word in words if len(word) > 1 and word not in stop_words]
    
    return words

class KnowledgeBaseOptimizer:
    def __init__(self, model="qwen-turbo-latest"):
        self.model = model
        self.knowledge_base = []
        self.content_bm25 = None
        self.question_bm25 = None
        self.content_documents = []
        self.question_documents = []
        self.content_metadata = []
        self.question_metadata = []
        
    def generate_questions_for_chunk(self, knowledge_chunk, num_questions=5):
        """为单个知识切片生成多样化问题"""
        instruction = """
你是一个专业的问答系统专家。给定的知识内容能回答哪些多样化的问题，这些问题可以：
1. 使用不同的问法（直接问、间接问、对比问等）
2. 避免重复和相似的问题
3. 确保问题不超出知识内容范围

请返回JSON格式：
{
    "questions": [
        {
            "question": "问题内容",
            "question_type": "问题类型（直接问/间接问/对比问/条件问等）",
            "difficulty": "难度等级（简单/中等/困难）"
        }
    ]
}
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 知识内容 ###
{knowledge_chunk}

### 生成问题数量 ###
{num_questions}

### 生成结果 ###
"""
        
        response = get_completion(prompt, self.model)
        
        # 预处理响应，移除markdown代码块格式
        response = preprocess_json_response(response)
        
        try:
            result = json.loads(response)
            return result.get('questions', [])
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            print(f"AI返回内容: {response[:50]}...")
            # 如果JSON解析失败，返回简单的问题列表
            return [{"question": f"关于{knowledge_chunk[:50]}...的问题", "question_type": "直接问", "keywords": [], "difficulty": "中等"}]
    
    def build_knowledge_index(self, knowledge_base):
        """构建知识库的BM25索引（包括原文和问题）"""
        print("正在构建知识库索引...")
        
        self.knowledge_base = knowledge_base
        # 初始化存储文档和元数据的列表
        # content_documents: 存储原文内容的分词结果
        # question_documents: 存储问题内容的分词结果  
        # content_metadata: 存储原文的元数据信息
        # question_metadata: 存储问题的元数据信息
        content_documents = []
        question_documents = []
        content_metadata = []
        question_metadata = []
        
        for i, chunk in enumerate(knowledge_base):
            # 获取知识切片的内容
            text = chunk.get('content', '')
            if not text.strip():
                continue
                
            # 原文文档
            content_words = preprocess_text(text)
            if content_words:
                content_documents.append(content_words)
                content_metadata.append({
                    "id": chunk.get('id', f"chunk_{i}"),
                    "content": text,
                    "category": chunk.get('category', ''),
                    "chunk": chunk,
                    "type": "content"
                })
            
            # 问题文档（如果存在生成的问题）
            if 'generated_questions' in chunk and chunk['generated_questions']:
                for j, question_data in enumerate(chunk['generated_questions']):
                    question = question_data.get('question', '')
                    if question.strip():
                        # 拼接全文和问题，保持上下文
                        combined_text = f"内容：{text} 问题：{question}"
                        question_words = preprocess_text(combined_text)
                        
                        if question_words:
                            question_documents.append(question_words)
                            question_metadata.append({
                                "id": f"{chunk.get('id', f'chunk_{i}')}_q{j}",
                                "content": question,
                                "combined_content": combined_text,
                                "category": chunk.get('category', ''),
                                "chunk": chunk,
                                "type": "question",
                                "question_data": question_data
                            })
        
        # 创建BM25索引
        if content_documents:
            # 使用BM25Okapi算法构建原文内容的检索索引
            # BM25是一种基于词频的排序算法,用于计算查询和文档的相关性
            # content_documents包含了所有文档的分词结果
            # 构建索引后可以用于后续的相似度计算和文档检索
            self.content_bm25 = BM25Okapi(content_documents)
            self.content_documents = content_documents
            self.content_metadata = content_metadata
            print(f"原文索引构建完成，共索引 {len(content_documents)} 个知识切片")
        
        if question_documents:
            self.question_bm25 = BM25Okapi(question_documents)
            self.question_documents = question_documents
            self.question_metadata = question_metadata
            print(f"问题索引构建完成，共索引 {len(question_documents)} 个问题")
        
        if not content_documents and not question_documents:
            print("没有有效的内容可以索引")
    
    def search_similar_chunks(self, query, k=3, search_type="content"):
        """使用BM25搜索相似的内容（原文或问题）"""
        if search_type == "content":
            if not self.content_bm25:
                return []
            bm25 = self.content_bm25
            metadata_store = self.content_metadata
        elif search_type == "question":
            if not self.question_bm25:
                return []
            bm25 = self.question_bm25
            print('question_bm25=', bm25)
            metadata_store = self.question_metadata
        else:
            return []
        
        try:
            # 预处理查询
            query_words = preprocess_text(query)
            if not query_words:
                return []
            
            # 搜索最相似的k个内容
            # 使用BM25算法计算查询词与所有文档的相关性得分
            # bm25.get_scores()方法会返回一个分数数组
            # 数组中每个元素对应一个文档与查询的相关性得分
            # 分数越高表示文档与查询越相关
            # query_words是经过分词的查询文本
            scores = bm25.get_scores(query_words)
            
            # 获取top-k结果
            # 1. np.argsort(scores) 对分数数组进行排序,返回排序后的索引数组
            # 2. [::-1] 将索引数组反转,使其按降序排列(分数从高到低)
            # 3. [:k] 获取前k个最高分数的索引
            # 例如:scores=[0.2, 0.8, 0.4], k=2
            # argsort结果为[1,2,0],反转后为[0,2,1],取前2个得到[0,2]
            top_indices = np.argsort(scores)[::-1][:k]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # 只返回有相关性的结果
                    metadata = metadata_store[idx]
                    # 将BM25分数转换为0-1范围的相似度
                    similarity = min(1.0, scores[idx] / 10.0)  # 归一化
                    results.append({
                        "metadata": metadata,
                        "score": scores[idx],
                        "similarity": similarity
                    })
            
            return results
            
        except Exception as e:
            print(f"搜索失败: {e}")
            return []
    
    def calculate_similarity(self, query, knowledge_chunk):
        """计算查询与知识切片的相似度（使用BM25）"""
        try:
            query_words = preprocess_text(query)
            chunk_words = preprocess_text(knowledge_chunk)
            
            if not query_words or not chunk_words:
                return 0.0
            
            # 创建临时BM25索引
            temp_bm25 = BM25Okapi([chunk_words])
            scores = temp_bm25.get_scores(query_words)
            
            # 返回最高分数并归一化
            max_score = max(scores) if scores else 0.0
            return min(1.0, max_score / 10.0)
            
        except Exception as e:
            print(f"相似度计算失败: {e}")
            return 0.0
    
    def calculate_question_similarity(self, user_query, generated_questions):
        """计算用户查询与生成问题的相似度"""
        similarities = []
        for question_data in generated_questions:
            question = question_data['question']
            similarity = self.calculate_similarity(user_query, question)
            similarities.append(similarity)
        return max(similarities) if similarities else 0.0
    
    def evaluate_retrieval_methods(self, knowledge_base, test_queries):
        """评估两种检索方法的准确度"""
        # 首先构建知识库索引（包括原文和问题）
        self.build_knowledge_index(knowledge_base)
        
        results = {
            'content_similarity': [],
            'question_similarity': [],
            'improvement': [],
            'content_scores': [],
            'question_scores': [],
            'query_details': []
        }
        
        for i, query_info in enumerate(test_queries):
            user_query = query_info['query']
            correct_chunk = query_info['correct_chunk']
            
            # 方法1：BM25原文检索
            content_results = self.search_similar_chunks(user_query, k=1, search_type="content")
            content_correct = False
            content_score = 0.0
            content_chunk_id = None
            if content_results:
                best_match = content_results[0]['metadata']['chunk']
                content_correct = best_match['content'] == correct_chunk
                content_score = content_results[0]['similarity']
                content_chunk_id = best_match['id']
            
            # 方法2：BM25问题检索
            question_results = self.search_similar_chunks(user_query, k=1, search_type="question")
            question_correct = False
            question_score = 0.0
            question_chunk_id = None
            if question_results:
                best_match = question_results[0]['metadata']['chunk']
                question_correct = best_match['content'] == correct_chunk
                question_score = question_results[0]['similarity']
                question_chunk_id = best_match['id']
            
            results['content_similarity'].append(content_correct)
            results['question_similarity'].append(question_correct)
            results['improvement'].append(question_correct and not content_correct)
            results['content_scores'].append(content_score)
            results['question_scores'].append(question_score)
            
            # 记录查询详情
            results['query_details'].append({
                'query': user_query,
                'content_score': content_score,
                'question_score': question_score,
                'content_correct': content_correct,
                'question_correct': question_correct,
                'score_diff': question_score - content_score,
                'content_chunk_id': content_chunk_id,
                'question_chunk_id': question_chunk_id
            })
        
        return results
    
    def generate_diverse_questions(self, knowledge_chunk, num_questions=8):
        """生成更多样化的问题（更丰富）"""
        instruction = """
你是一个专业的问答系统专家。请为给定的知识内容生成高度多样化的问题，确保：
1. 问题类型多样化：直接问、间接问、对比问、条件问、假设问、推理问等
2. 表达方式多样化：使用不同的句式、词汇、语气
3. 难度层次多样化：简单、中等、困难的问题都要有
4. 角度多样化：从不同角度和维度提问
5. 确保问题不超出知识内容范围

请返回JSON格式：
{
    "questions": [
        {
            "question": "问题内容",
            "question_type": "问题类型",
            "difficulty": "难度等级",
            "perspective": "提问角度",
            "is_answerable": "给出的知识能否回答该问题",
            "answer": "基于该知识的回答"
        }
    ]
}
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 知识内容 ###
{knowledge_chunk}

### 生成问题数量 ###
{num_questions}

### 生成结果 ###
"""
        
        response = get_completion(prompt, self.model)
        
        # 预处理响应，移除markdown代码块格式
        response = preprocess_json_response(response)
        
        try:
            result = json.loads(response)
            return result.get('questions', [])
        except json.JSONDecodeError as e:
            print(f"多样化问题生成JSON解析失败: {e}")
            print(f"AI返回内容: {response[:200]}...")
            return []

def main():
    # 初始化知识库优化器
    optimizer = KnowledgeBaseOptimizer()
    
    print("=== 知识库问题生成与检索优化示例（BM25版本）- 迪士尼主题乐园 ===\n")
    
    # 示例知识库
    knowledge_base = [
        {
            "id": "kb_001",
            "content": "上海迪士尼乐园位于上海市浦东新区，是中国大陆首座迪士尼主题乐园，于2016年6月16日开园。乐园占地面积390公顷，包含七大主题园区：米奇大街、奇想花园、探险岛、宝藏湾、明日世界、梦幻世界和迪士尼小镇。",
            "category": "基本信息"
        },
        {
            "id": "kb_002", 
            "content": "上海迪士尼乐园的门票价格根据季节和日期有所不同。平日成人票价为399元，周末和节假日为499元。儿童票（1.0-1.4米）平日为299元，周末和节假日为374元。1.0米以下儿童免费入园。",
            "category": "价格信息"
        },
        {
            "id": "kb_003",
            "content": "上海迪士尼乐园的营业时间通常为上午8:00至晚上8:00，但具体时间会根据季节和特殊活动进行调整。建议游客在出发前查看官方网站或APP获取最新的营业时间信息。",
            "category": "营业信息"
        },
        {
            "id": "kb_004",
            "content": "从上海市区到上海迪士尼乐园有多种交通方式：1. 地铁11号线迪士尼站下车；2. 乘坐迪士尼专线巴士；3. 打车约40-60分钟；4. 自驾车可停在乐园停车场，停车费为100元/天。",
            "category": "交通信息"
        },
        {
            "id": "kb_005",
            "content": "上海迪士尼乐园的特色项目包括：创极速光轮（明日世界）、七个小矮人矿山车（梦幻世界）、加勒比海盗：战争之潮（宝藏湾）、翱翔·飞越地平线（探险岛）等。这些项目都有不同的身高和年龄限制。",
            "category": "游乐项目"
        },
        {
            "id": "kb_006",
            "content": "上海迪士尼乐园提供多种餐饮选择，包括米奇大街的皇家宴会厅、奇想花园的漫月轩、宝藏湾的巴波萨烧烤等。园内餐厅价格相对较高，人均消费约150-300元。建议游客可以携带密封包装的零食和水入园。",
            "category": "餐饮信息"
        },
        {
            "id": "kb_007",
            "content": "上海迪士尼乐园的购物体验非常丰富，每个主题园区都有特色商店。米奇大街的M大街购物廊是最大的综合商店，销售各种迪士尼周边商品。建议游客在离园前购买纪念品，避免携带不便。",
            "category": "购物信息"
        },
        {
            "id": "kb_008",
            "content": "上海迪士尼乐园提供多种服务设施，包括婴儿车租赁（50元/天）、轮椅租赁（免费）、储物柜（60元/天）、充电宝租赁等。园内设有多个医疗点和失物招领处，为游客提供便利服务。",
            "category": "服务设施"
        }
    ]
    
    # 示例1: 为知识切片生成问题
    print("示例1: 为知识切片生成多样化问题")
    test_chunk = knowledge_base[0]['content']
    print(f"知识内容: {test_chunk}")
    
    # doc2query
    questions = optimizer.generate_questions_for_chunk(test_chunk, num_questions=5)
    print(f"\n生成的5个问题:")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q['question']} (类型: {q['question_type']}, 难度: {q['difficulty']})")
    
    print("\n" + "="*60 + "\n")
    
    # doc2query
    # 示例2: 生成更多样化的问题
    print("示例2: 生成更多样化的问题（8个）")
    diverse_questions = optimizer.generate_diverse_questions(test_chunk, num_questions=8)
    print(f"\n生成的8个多样化问题:")
    for i, q in enumerate(diverse_questions, 1):
        print(f"  {i}. {q['question']}")
        print(f"     类型: {q['question_type']}, 难度: {q['difficulty']}, 角度: {q['perspective']}, 能否回答: {q['is_answerable']}, 回答的答案：{q['answer']}")
    
    print("\n" + "="*60 + "\n")
    
    # 示例3: 评估检索方法
    print("示例3: 评估两种检索方法的准确度")
    
    # 测试查询 - 设计更有挑战性的问题
    test_queries = [
        {
            "query": "如果我想体验最刺激的过山车，应该去哪个区域？",
            "correct_chunk": knowledge_base[4]['content']
        },
        {
            "query": "什么时间去人比较少？",
            "correct_chunk": knowledge_base[2]['content']
        },
        {
            "query": "可以带食物进去吗？",
            "correct_chunk": knowledge_base[5]['content']
        }
    ]
    
    # 为知识库生成问题
    print('正在为知识库生成问题...')
    for chunk in knowledge_base:
        chunk['generated_questions'] = optimizer.generate_questions_for_chunk(chunk['content'])
        #print("chunk['generated_questions']=", chunk['generated_questions'])
    print('为知识库生成问题完毕')
    
    # 评估检索方法
    results = optimizer.evaluate_retrieval_methods(knowledge_base, test_queries)
    
    print(f"测试查询数量: {len(test_queries)}")
    print(f"BM25原文检索准确率: {sum(results['content_similarity'])/len(results['content_similarity'])*100:.1f}%")
    print(f"BM25问题检索准确率: {sum(results['question_similarity'])/len(results['question_similarity'])*100:.1f}%")
    print(f"问题检索改进的查询数量: {sum(results['improvement'])}")
    
    # 详细分析
    print(f"\n=== 详细分析 ===")
    
    # 按相似度分数差异排序
    # 对查询结果按照分数差异(score_diff)进行排序
    # score_diff 表示问题检索分数与原文检索分数的差值
    # reverse=True 表示按降序排序,即分数差异大的排在前面
    # 这样可以看出哪些查询在问题检索方法上表现更好(score_diff为正)
    # 哪些查询在原文检索方法上表现更好(score_diff为负)
    sorted_details = sorted(results['query_details'], key=lambda x: x['score_diff'], reverse=True)
    
    print(f"\n问题检索方法表现更好的查询（按分数差异排序）:")
    # 遍历前5个检索结果详情,使用enumerate生成序号(从1开始)
    # sorted_details[:5]表示取排序后的前5条记录
    # i是序号(1-5), detail是每条检索结果的详细信息
    # enumerate是一个内置函数,用于同时获取元素索引和元素值
    # 这里的enumerate(sorted_details[:5], 1)表示从sorted_details[:5]的第一个元素开始,索引从1开始计数
    for i, detail in enumerate(sorted_details[:5], 1):
        if detail['score_diff'] > 0:
            print(f"  {i}. 查询: {detail['query']}")
            print(f"     原文检索分数: {detail['content_score']:.3f}")
            print(f"     问题检索分数: {detail['question_score']:.3f}")
            print(f"     分数差异: +{detail['score_diff']:.3f}")
            print(f"     原文检索: {'✓' if detail['content_correct'] else '✗'}")
            print(f"     问题检索: {'✓' if detail['question_correct'] else '✗'}")
    
    print(f"\n原文检索方法表现更好的查询:")
    for i, detail in enumerate(sorted_details[-5:], 1):
        if detail['score_diff'] < 0:
            print(f"  {i}. 查询: {detail['query']}")
            print(f"     原文检索分数: {detail['content_score']:.3f}")
            print(f"     问题检索分数: {detail['question_score']:.3f}")
            print(f"     分数差异: {detail['score_diff']:.3f}")
            print(f"     原文检索: {'✓' if detail['content_correct'] else '✗'}")
            print(f"     问题检索: {'✓' if detail['question_correct'] else '✗'}")
        

if __name__ == "__main__":
    main() 