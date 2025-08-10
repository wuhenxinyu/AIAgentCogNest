# 对话知识提取与沉淀
"""
产品上线后每天产生大量对话，如何从这些对话中提取和沉淀有价值的知识，持续丰富知识库？
TO DO：对话知识沉淀
    核心功能：
    • 使用AI模型（通义千问）从对话中提取结构化知识
    • 支持多种知识类型：事实、问题、流程、注意事项等
    • 自动识别用户意图和对话摘要
    核心函数：
    • extract_knowledge_from_conversation() : 从单次对话中提取知识
    • batch_extract_knowledge() : 批量提取知识
    • merge_similar_knowledge()：使用LLM合并相似知识点
"""
# 导入依赖库
import dashscope
import os
import json
from datetime import datetime
from collections import Counter
os.environ['DASHSCOPE_API_KEY'] = 'sk-58f051ae745e4bb19fdca31735105b11'
# 从环境变量中获取 API Key
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

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
    response = dashscope.Generation.call(
        model=model,
        messages=messages,
        result_format='message',
        temperature=0.3,
    )
    return response.output.choices[0].message.content

class ConversationKnowledgeExtractor:
    def __init__(self, model="qwen-turbo-latest"):
        self.model = model
        self.extracted_knowledge = []
        # 使用Counter类来统计知识点出现的频率
        # 键格式为"知识类型:知识内容前50个字符"
        # 值为该知识点出现的次数
        self.knowledge_frequency = Counter()
        
    def extract_knowledge_from_conversation(self, conversation):
        """从单次对话中提取知识"""
        instruction = """
你是一个专业的知识提取专家。请从给定的对话中提取有价值的知识点，包括：
1. 事实性信息（地点、时间、价格、规则等）
2. 用户需求和偏好
3. 常见问题和解答
4. 操作流程和步骤
5. 注意事项和提醒

请返回JSON格式：
{
    "extracted_knowledge": [
        {
            "knowledge_type": "知识类型（事实/需求/问题/流程/注意）",
            "content": "知识内容",
            "confidence": "置信度(0-1)",
            "source": "来源（用户/AI/对话）",
            "keywords": ["关键词1", "关键词2"],
            "category": "分类"
        }
    ],
    "conversation_summary": "对话摘要",
    "user_intent": "用户意图"
}
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 对话内容 ###
{conversation}

### 提取结果 ###
"""
        
        response = get_completion(prompt, self.model)
        
        # 预处理响应，移除markdown代码块格式
        response = preprocess_json_response(response)
        
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError as e:
            print(f"对话知识提取JSON解析失败: {e}")
            print(f"AI返回内容: {response[:200]}...")
            return {
                "extracted_knowledge": [],
                "conversation_summary": "无法解析对话",
                "user_intent": "未知"
            }
    
    def batch_extract_knowledge(self, conversations):
        """批量提取知识"""
        all_knowledge = []
        
        for i, conversation in enumerate(conversations):
            print(f"正在处理对话 {i+1}/{len(conversations)}...")
            
            result = self.extract_knowledge_from_conversation(conversation)
            all_knowledge.extend(result.get('extracted_knowledge', []))
            
            # 更新频率统计
            for knowledge in result.get('extracted_knowledge', []):
                key = f"{knowledge['knowledge_type']}:{knowledge['content'][:50]}"
                self.knowledge_frequency[key] += 1
        
        return all_knowledge
    
    def merge_similar_knowledge(self, knowledge_list):
        """使用LLM合并相似的知识点，过滤掉需求和问题类型"""
        # 过滤掉需求和问题类型的知识，因为它们是临时的、个性化的
        filtered_knowledge = [
            knowledge for knowledge in knowledge_list 
            if knowledge.get('knowledge_type') not in ['需求', '问题']
        ]
        
        print(f"过滤前知识点数量: {len(knowledge_list)}")
        print(f"过滤后知识点数量: {len(filtered_knowledge)}")
        print(f"过滤掉的'需求'和'问题'类型知识点: {len(knowledge_list) - len(filtered_knowledge)}")
        
        # 按知识类型分组
        knowledge_by_type = {}
        for knowledge in filtered_knowledge:
            knowledge_type = knowledge.get('knowledge_type', '其他')
            if knowledge_type not in knowledge_by_type:
                knowledge_by_type[knowledge_type] = []
            knowledge_by_type[knowledge_type].append(knowledge)
        
        merged_knowledge = []
        
        # 对每个知识类型分别进行LLM合并
        for knowledge_type, knowledge_group in knowledge_by_type.items():
            if len(knowledge_group) == 1:
                # 只有一个知识点，直接添加
                merged_knowledge.append(knowledge_group[0])
            else:
                # 多个知识点，使用LLM合并
                merged = self.merge_knowledge_with_llm(knowledge_group, knowledge_type)
                merged_knowledge.append(merged)
        
        return merged_knowledge
    
    def merge_knowledge_with_llm(self, knowledge_group, knowledge_type):
        """使用LLM合并同类型的知识组"""
        # 准备知识内容列表
        knowledge_contents = []
        all_keywords = set()
        all_sources = []
        
        for i, knowledge in enumerate(knowledge_group, 1):
            content = knowledge.get('content', '')
            confidence = knowledge.get('confidence', 0.5)
            keywords = knowledge.get('keywords', [])
            source = knowledge.get('source', '')
            category = knowledge.get('category', '')
            
            knowledge_contents.append(f"{i}. 内容: {content}")
            knowledge_contents.append(f"   置信度: {confidence}")
            knowledge_contents.append(f"   分类: {category}")
            knowledge_contents.append(f"   来源: {source}")
            knowledge_contents.append(f"   关键词: {', '.join(keywords)}")
            knowledge_contents.append("")
            
            all_keywords.update(keywords)
            if source and source not in all_sources:
                all_sources.append(source)
        
        # 构建LLM合并提示
        prompt = f"""
你是一个专业的知识整理专家。请将以下{knowledge_type}类型的知识点进行智能合并，生成一个更完整、准确的知识点。

### 合并要求：
1. 保留所有重要信息，避免信息丢失
2. 消除重复内容，整合相似表述
3. 提高内容的准确性和完整性
4. 保持逻辑清晰，结构合理
5. 合并后的置信度取所有知识点中的最高值

### 待合并的知识点：
{chr(10).join(knowledge_contents)}

### 请返回JSON格式：
{{
    "knowledge_type": "{knowledge_type}",
    "content": "合并后的知识内容",
    "confidence": 最高置信度值,
    "keywords": ["合并后的关键词列表"],
    "category": "合并后的分类",
    "sources": ["所有来源"],
    "frequency": {len(knowledge_group)}
}}

### 合并结果：
"""
        
        response = get_completion(prompt, self.model)
        
        # 预处理响应
        response = preprocess_json_response(response)
        
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError as e:
            print(f"知识合并JSON解析失败: {e}")
            print(f"AI返回内容: {response[:200]}...")
            
            # 解析失败时，使用简单的合并策略
            best_knowledge = max(knowledge_group, key=lambda x: x.get('confidence', 0))
            return {
                "knowledge_type": knowledge_type,
                "content": best_knowledge['content'],
                "confidence": best_knowledge.get('confidence', 0.5),
                "frequency": len(knowledge_group),
                "keywords": list(all_keywords),
                "category": best_knowledge['category'],
                "sources": all_sources
            }
def main():
    # 初始化对话知识提取器
    extractor = ConversationKnowledgeExtractor()
    
    print("=== 对话知识提取与沉淀示例（迪士尼主题乐园） ===\n")
    
    # 示例对话数据
    sample_conversations = [
        """
用户: "我想去上海迪士尼乐园玩，门票多少钱？"
AI: "上海迪士尼乐园的门票价格根据日期有所不同。平日成人票价为399元，周末和节假日为499元。儿童票（1.0-1.4米）平日为299元，周末为374元。1.0米以下儿童免费。"
用户: "需要提前预订吗？"
AI: "建议提前预订，特别是周末和节假日，可以通过官方网站或第三方平台预订。"
用户: "从浦东机场怎么去？"
AI: "从浦东机场到迪士尼乐园可以乘坐地铁2号线到广兰路站，然后换乘11号线到迪士尼站，全程约1小时。也可以打车，约40分钟。"
""",
        """
用户: "迪士尼乐园今天开放吗？"
AI: "上海迪士尼乐园通常每天开放，营业时间为上午8:00至晚上8:00。但建议您出发前查看官方网站或APP确认具体营业时间，因为可能会因特殊活动或维护调整。"
用户: "现在人多不多？"
AI: "人流量会因日期、季节和活动而变化。周末、节假日和寒暑假通常人较多。建议您查看官方APP的实时人流量信息，或关注社交媒体上的游客分享。"
用户: "有什么必玩的项目？"
AI: "上海迪士尼乐园的必玩项目包括：创极速光轮（明日世界）、七个小矮人矿山车（梦幻世界）、加勒比海盗：战争之潮（宝藏湾）、翱翔·飞越地平线（探险岛）等。"
""",
        """
用户: "带小孩去迪士尼需要注意什么？"
AI: "带小孩去迪士尼需要注意：1. 身高限制，部分项目有身高要求；2. 准备防晒、雨具等；3. 带些零食和水；4. 建议下载官方APP查看实时排队时间；5. 可以租用婴儿车；6. 注意安全，看管好孩子。"
用户: "停车费怎么收？"
AI: "上海迪士尼乐园停车场收费为100元/天，建议提前到达，因为停车场可能会满。也可以选择地铁出行，11号线迪士尼站就在乐园门口。"
用户: "可以带食物进去吗？"
AI: "可以携带密封包装的零食和水，但不能带玻璃瓶、酒精饮料等。园内有多个餐厅和小吃店，价格相对较高，建议合理安排。"
"""
    ]
    
    # 示例1: 从单次对话中提取知识
    print("示例1: 从单次对话中提取知识")
    conversation = sample_conversations[0]
    print(f"对话内容:\n{conversation}")
    
    extracted = extractor.extract_knowledge_from_conversation(conversation)
    print(f"\n提取的知识点:")
    for i, knowledge in enumerate(extracted['extracted_knowledge'], 1):
        print(f"  {i}. 类型: {knowledge['knowledge_type']}")
        print(f"     内容: {knowledge['content']}")
        print(f"     置信度: {knowledge['confidence']}")
        print(f"     分类: {knowledge['category']}")
    
    print(f"\n对话摘要: {extracted['conversation_summary']}")
    print(f"用户意图: {extracted['user_intent']}")
    
    print("\n" + "="*60 + "\n")
    
    # 示例2: 批量提取知识
    print("示例2: 批量提取知识")
    all_knowledge = extractor.batch_extract_knowledge(sample_conversations)
    print(f"总共提取了 {len(all_knowledge)} 个知识点")
    
    # 显示所有知识点
    print(f"\n所有知识点:")
    for key, count in extractor.knowledge_frequency.most_common():
        print(f"  {key}: {count}次")
    
    print("\n" + "="*60 + "\n")
    
    
    # 示例3: 合并相似知识
    print("示例3: 合并相似知识")
    merged_knowledge = extractor.merge_similar_knowledge(all_knowledge)
    print(f"合并后剩余 {len(merged_knowledge)} 个知识点")
    
    print(f"\n合并后的知识点:")
    for i, knowledge in enumerate(merged_knowledge, 1):
        print(f"  {i}. 类型: {knowledge.get('knowledge_type', '未知')}")
        print(f"     内容: {knowledge['content']}")
        print(f"     频率: {knowledge.get('frequency', 1)}次")
        print(f"     置信度: {knowledge.get('confidence', 0.5)}")
        print(f"     分类: {knowledge.get('category', '未知')}")
        print(f"     关键词: {knowledge.get('keywords', [])}")
        print(f"     来源: {knowledge.get('sources', [])}")
        print()
    
    print("\n" + "="*60 + "\n")    

if __name__ == "__main__":
    main() 