# Query改写使用示例
"""
Query改写是指将用户的原始查询语句转换为一个或多个新的查询语句，以提高检索的准确性和召回率。
Query改写的类型：
1. 上下文依赖型Query改写
2. 对比型Query改写
3. 模糊指代型Query改写
4. 多意图型Query改写
5. 反问型Query改写
6. 自动识别Query类型并进行改写（意图识别）
"""
# 导入依赖库
import dashscope
import os
import json

os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here'
# 从环境变量中获取 API Key
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

# 基于 prompt 生成文本
def get_completion(prompt, model="qwen-turbo-latest"):
    messages = [{"role": "user", "content": prompt}]
    response = dashscope.Generation.call(
        model=model,
        messages=messages,
        result_format='message',
        temperature=0.5, # 温度参数，控制生成文本的随机性，默认值为0.5，取值范围为[0, 1]，值越大，生成的文本越随机
        top_p=0.8, #  nucleus sampling 参数，控制生成文本的多样性，默认值为0.8，取值范围为[0, 1]，值越大，生成的文本越多样
        presence_penalty=0.0, # 存在惩罚参数，控制生成文本中出现重复内容的程度，默认值为0.0，取值范围为[0, 1]，值越大，生成的文本中重复内容越少
        frequency_penalty=0.0, # 频率惩罚参数，控制生成文本中出现频繁内容的程度，默认值为0.0，取值范围为[0, 1]，值越大，生成的文本中频繁内容越少
    )
    return response.output.choices[0].message.content


# Query改写功能
"""
Query改写的功能：
1. 上下文依赖型Query改写
2. 对比型Query改写
3. 模糊指代型Query改写
4. 多意图型Query改写
5. 反问型Query改写
6. 自动识别Query类型并进行改写（意图识别）

"""
class QueryRewriter:
    def __init__(self, model="qwen-turbo-latest"):
        self.model = model
    
    def rewrite_context_dependent_query(self, current_query, conversation_history):
        """上下文依赖型Query改写"""
        instruction = """
你是一个智能的查询优化助手。请分析用户的当前问题以及前序对话历史，判断当前问题是否依赖于上下文。
如果依赖，请将当前问题改写成一个独立的、包含所有必要上下文信息的完整问题。
如果不依赖，直接返回原问题。
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history}

### 当前问题 ###
{current_query}

### 改写后的问题 ###
"""
        
        return get_completion(prompt, self.model)
    


    def rewrite_comparative_query(self, query, context_info):
        """对比型Query改写"""
        instruction = """
你是一个查询分析专家。请分析用户的输入和相关的对话上下文，识别出问题中需要进行比较的多个对象。
然后，将原始问题改写成一个更明确、更适合在知识库中检索的对比性查询。
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 对话历史/上下文信息 ###
{context_info}

### 原始问题 ###
{query}

### 改写后的查询 ###
"""
        
        return get_completion(prompt, self.model)
    


    def rewrite_ambiguous_reference_query(self, current_query, conversation_history):
        """模糊指代型Query改写"""
        instruction = """
你是一个消除语言歧义的专家。请分析用户的当前问题和对话历史，找出问题中 "都"、"它"、"这个" 等模糊指代词具体指向的对象。
然后，将这些指代词替换为明确的对象名称，生成一个清晰、无歧义的新问题。
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history}

### 当前问题 ###
{current_query}

### 改写后的问题 ###
"""
        
        return get_completion(prompt, self.model)
    

    def rewrite_multi_intent_query(self, query):
        """多意图型Query改写 - 分解查询"""
        instruction = """
你是一个任务分解机器人。请将用户的复杂问题分解成多个独立的、可以单独回答的简单问题。以JSON数组格式输出。
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 原始问题 ###
{query}

### 分解后的问题列表 ###
请以JSON数组格式输出，例如：["问题1", "问题2", "问题3"]
"""
        
        response = get_completion(prompt, self.model)
        try:
            return json.loads(response)
        except:
            return [response]
    

    def rewrite_rhetorical_query(self, current_query, conversation_history):
        """反问型Query改写"""
        instruction = """
你是一个沟通理解大师。请分析用户的反问或带有情绪的陈述，识别其背后真实的意图和问题。
然后，将这个反问改写成一个中立、客观、可以直接用于知识库检索的问题。
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history}

### 当前问题 ###
{current_query}

### 改写后的问题 ###
"""
        
        return get_completion(prompt, self.model)
    


    def auto_rewrite_query(self, query, conversation_history="", context_info=""):
        """自动识别Query类型并进行改写（意图识别）"""
        instruction = """
你是一个智能的查询分析专家。请分析用户的查询，识别其属于以下哪种类型：
1. 上下文依赖型 - 包含"还有"、"其他"等需要上下文理解的词汇
2. 对比型 - 包含"哪个"、"比较"、"更"、"哪个更好"、"哪个更"等比较词汇
3. 模糊指代型 - 包含"它"、"他们"、"都"、"这个"等指代词
4. 多意图型 - 包含多个独立问题，用"、"或"？"分隔
5. 反问型 - 包含"不会"、"难道"等反问语气
说明：如果同时存在多意图型、模糊指代型，优先级为多意图型>模糊指代型

请返回JSON格式的结果：
{
    "query_type": "查询类型",
    "rewritten_query": "改写后的查询",
    "confidence": "置信度(0-1)"
}
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history}

### 上下文信息 ###
{context_info}

### 原始查询 ###
{query}

### 分析结果 ###
"""
        
        response = get_completion(prompt, self.model)
        try:
            return json.loads(response)
        except:
            return {
                "query_type": "未知类型",
                "rewritten_query": query,
                "confidence": 0.5
            }
    
    def auto_rewrite_and_execute(self, query, conversation_history="", context_info=""):
        """自动识别Query类型并进行改写，然后根据类型调用相应的改写方法"""
        # 首先进行自动识别
        result = self.auto_rewrite_query(query, conversation_history, context_info)
        
        # 根据识别结果调用相应的改写方法
        query_type = result.get('query_type', '')
        
        if '上下文依赖' in query_type:
            final_result = self.rewrite_context_dependent_query(query, conversation_history)
        elif '对比' in query_type:
            final_result = self.rewrite_comparative_query(query, context_info or conversation_history)
        elif '模糊指代' in query_type:
            final_result = self.rewrite_ambiguous_reference_query(query, conversation_history)
        elif '多意图' in query_type:
            final_result = self.rewrite_multi_intent_query(query)
        elif '反问' in query_type:
            final_result = self.rewrite_rhetorical_query(query, conversation_history)
        else:
            # 对于其他类型，返回自动识别的改写结果
            final_result = result.get('rewritten_query', query)
        
        return {
            "original_query": query,
            "detected_type": query_type,
            "confidence": result.get('confidence', 0.5),
            "rewritten_query": final_result,
            "auto_rewrite_result": result
        }

def main():
    # 初始化Query改写器
    rewriter = QueryRewriter()    
    print("=== Query改写功能使用示例（迪士尼主题乐园） ===\n")
    
    # 示例1: 上下文依赖型Query
    print("示例1: 上下文依赖型Query")
    conversation_history = """
用户: "我想了解一下上海迪士尼乐园的最新项目。"
AI: "上海迪士尼乐园最新推出了'疯狂动物城'主题园区，这里有朱迪警官和尼克狐的互动体验。"
用户: "这个园区有什么游乐设施？"
AI: "'疯狂动物城'园区目前有疯狂动物城警察局、朱迪警官训练营和尼克狐的冰淇淋店等设施。"
"""
    current_query = "还有其他设施吗？"
    
    print(f"对话历史: {conversation_history}")
    print(f"当前查询: {current_query}")
    
    result = rewriter.rewrite_context_dependent_query(current_query, conversation_history)
    print(f"改写结果: {result}\n")
    
    # 示例2: 对比型Query
    print("示例2: 对比型Query")
    conversation_history = """
用户: "我想了解一下上海迪士尼乐园的最新项目。"
AI: "上海迪士尼乐园最新推出了疯狂动物城主题园区，还有蜘蛛侠主题园区"
"""
    current_query = "哪个游玩的时间比较长，比较有趣"
    
    print(f"对话历史: {conversation_history}")
    print(f"当前查询: {current_query}")
    
    result = rewriter.rewrite_comparative_query(current_query, conversation_history)
    print(f"改写结果: {result}\n")
    
    # 示例3: 模糊指代型Query
    print("示例3: 模糊指代型Query")
    conversation_history = """
用户: "我想了解一下上海迪士尼乐园和香港迪士尼乐园的烟花表演。"
AI: "好的，上海迪士尼乐园和香港迪士尼乐园都有精彩的烟花表演。"
"""
    current_query = "都什么时候开始？"
    
    print(f"对话历史: {conversation_history}")
    print(f"当前查询: {current_query}")
    
    result = rewriter.rewrite_ambiguous_reference_query(current_query, conversation_history)
    print(f"改写结果: {result}\n")
    
    # 示例4: 多意图型Query
    print("示例4: 多意图型Query")
    query = "门票多少钱？需要提前预约吗？停车费怎么收？"
    
    print(f"原始查询: {query}")
    
    result = rewriter.rewrite_multi_intent_query(query)
    print(f"分解结果: {result}\n")
    
    # 示例5: 反问型Query
    print("示例5: 反问型Query")
    conversation_history = """
用户: "你好，我想预订下周六上海迪士尼乐园的门票。"
AI: "正在为您查询... 查询到下周六的门票已经售罄。"
用户: "售罄是什么意思？我朋友上周去还能买到当天的票。"
"""
    current_query = "这不会也要提前一个月预订吧？"
    
    print(f"对话历史: {conversation_history}")
    print(f"当前查询: {current_query}")
    
    result = rewriter.rewrite_rhetorical_query(current_query, conversation_history)
    print(f"改写结果: {result}\n")
    
    # 示例6: 自动识别Query类型
    print("示例6: 自动识别Query类型")
    test_queries = [
        "还有其他游乐项目吗？",
        "哪个园区更好玩？",
        "都适合小朋友吗？",
        "有什么餐厅？价格怎么样？",
        "这不会也要排队两小时吧？"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"测试查询 {i}: {query}")
        result = rewriter.auto_rewrite_query(query)
        print(f"  识别类型: {result['query_type']}")
        print(f"  改写结果: {result['rewritten_query']}")
        print(f"  置信度: {result['confidence']}\n")

if __name__ == "__main__":
    main() 