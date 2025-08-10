# 知识库健康度检查
"""
如何对整个知识库进行健康度检查，找出缺少的知识、过期的知识、冲突的知识，确保知识库的质量和可靠性。
TO DO：知识库健康度检查
    核心功能：
    • 完整性检查：评估知识库是否覆盖用户的主要查询需求
    • 时效性检查：识别过期或需要更新的知识内容
    • 一致性检查：发现知识库中的冲突和矛盾信息
    • 综合评分：提供量化的健康度评分和改进建议
"""
# 导入依赖库
import dashscope
import os
import json
import re
from datetime import datetime

# 从环境变量中获取 API Key
os.environ['DASHSCOPE_API_KEY'] = 'sk-58f051ae745e4bb19fdca31735105b11'
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

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

class KnowledgeBaseHealthChecker:
    def __init__(self, model="qwen-turbo-latest"):
        self.model = model
        self.health_report = {}
        
    def check_missing_knowledge(self, knowledge_base, test_queries):
        """使用LLM检查缺少的知识"""
        instruction = """
你是一个知识库完整性检查专家。请分析给定的测试查询和知识库内容，判断知识库中是否缺少相关的知识。

检查标准：
1. 查询是否能在知识库中找到相关答案
2. 知识是否完整、准确
3. 是否覆盖了用户的主要需求
4. 是否存在知识空白

请返回JSON格式：
{
    "missing_knowledge": [
        {
            "query": "测试查询",
            "missing_aspect": "缺少的知识方面",
            "importance": "重要性（高/中/低）",
            "suggested_content": "建议的知识内容",
            "category": "知识分类"
        }
    ],
    "coverage_score": "覆盖率评分(0-1)",
    "completeness_analysis": "完整性分析"
}
"""
        
        # 构建知识库内容摘要
        knowledge_summary = []
        for chunk in knowledge_base:
            knowledge_summary.append(f"ID: {chunk.get('id', 'unknown')} - {chunk.get('content', '')}")
        
        knowledge_text = "\n".join(knowledge_summary)
        
        # 构建测试查询列表
        queries_text = []
        for query_info in test_queries:
            query = query_info['query']
            expected = query_info.get('expected_answer', '')
            queries_text.append(f"查询: {query} | 期望答案: {expected}")
        
        queries_text = "\n".join(queries_text)
        
        prompt = f"""
### 指令 ###
{instruction}

### 知识库内容 ###
{knowledge_text}

### 测试查询 ###
{queries_text}

### 分析结果 ###
"""
        
        try:
            response = get_completion(prompt, self.model)
            
            # 预处理响应，移除markdown代码块格式
            if response.startswith('```json'):
                response = response[7:]
            elif response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            
            result = json.loads(response.strip())
            return result
            
        except Exception as e:
            print(f"LLM检查缺少知识失败: {e}")
            return None
    
    def check_outdated_knowledge(self, knowledge_base):
        """使用LLM检查过期的知识"""
        instruction = """
你是一个知识时效性检查专家。请分析给定的知识内容，判断是否存在过期或需要更新的信息。

检查标准：
1. 时间相关信息是否过期（年份、日期、时间范围）
2. 价格信息是否最新（价格、费用、票价等）
3. 政策规则是否更新（政策、规定、规则等）
4. 活动信息是否有效（活动、节日、特殊安排等）
5. 联系方式是否准确（电话、地址、网址等）
6. 技术信息是否过时（版本、技术标准等）

请返回JSON格式：
{
    "outdated_knowledge": [
        {
            "chunk_id": "知识切片ID",
            "content": "知识内容",
            "outdated_aspect": "过期方面",
            "severity": "严重程度（高/中/低）",
            "suggested_update": "建议更新内容",
            "last_verified": "最后验证时间"
        }
    ],
    "freshness_score": "新鲜度评分(0-1)",
    "update_recommendations": "更新建议"
}
"""
        
        # 构建知识库内容
        knowledge_text = []
        for chunk in knowledge_base:
            content = chunk.get('content', '')
            chunk_id = chunk.get('id', 'unknown')
            last_updated = chunk.get('last_updated', 'unknown')
            knowledge_text.append(f"ID: {chunk_id} | 更新时间: {last_updated} | 内容: {content}")
        
        knowledge_text = "\n".join(knowledge_text)
        
        prompt = f"""
### 指令 ###
{instruction}

### 知识库内容 ###
{knowledge_text}

### 当前时间 ###
{datetime.now().strftime('%Y年%m月%d日')}

### 分析结果 ###
"""
        
        try:
            response = get_completion(prompt, self.model)
            
            # 预处理响应，移除markdown代码块格式
            if response.startswith('```json'):
                response = response[7:]
            elif response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            
            result = json.loads(response.strip())
            return result
            
        except Exception as e:
            print(f"LLM检查过期知识失败: {e}")
            return None

            
    def check_conflicting_knowledge(self, knowledge_base):
        """使用LLM检查冲突的知识"""
        instruction = """
你是一个知识一致性检查专家。请分析给定的知识库，找出可能存在冲突或矛盾的信息。

检查标准：
1. 同一主题的不同说法（地点、名称、描述等）
2. 价格信息的差异（价格、费用、收费标准等）
3. 时间信息的不一致（营业时间、开放时间、活动时间等）
4. 规则政策的冲突（规定、政策、要求等）
5. 操作流程的差异（步骤、方法、流程等）
6. 联系方式的差异（地址、电话、网址等）

请返回JSON格式：
{
    "conflicting_knowledge": [
        {
            "conflict_type": "冲突类型",
            "chunk_ids": ["相关切片ID"],
            "conflicting_content": ["冲突内容"],
            "severity": "严重程度（高/中/低）",
            "resolution_suggestion": "解决建议"
        }
    ],
    "consistency_score": "一致性评分(0-1)",
    "conflict_analysis": "冲突分析"
}
"""
        
        # 构建知识库内容
        knowledge_text = []
        for chunk in knowledge_base:
            content = chunk.get('content', '')
            chunk_id = chunk.get('id', 'unknown')
            knowledge_text.append(f"ID: {chunk_id} | 内容: {content}")
        
        knowledge_text = "\n".join(knowledge_text)
        
        prompt = f"""
### 指令 ###
{instruction}

### 知识库内容 ###
{knowledge_text}

### 分析结果 ###
"""
        
        try:
            response = get_completion(prompt, self.model)
            
            # 预处理响应，移除markdown代码块格式
            if response.startswith('```json'):
                response = response[7:]
            elif response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            
            result = json.loads(response.strip())
            return result
            
        except Exception as e:
            print(f"LLM检查冲突知识失败: {e}")
            return None
    
    def calculate_overall_health_score(self, missing_result, outdated_result, conflicting_result):
        """计算整体健康度评分"""
        coverage_score = missing_result.get('coverage_score', 0)
        freshness_score = outdated_result.get('freshness_score', 0)
        consistency_score = conflicting_result.get('consistency_score', 0)
        
        # 加权计算
        overall_score = (
            coverage_score * 0.4 +      # 覆盖率权重40%
            freshness_score * 0.3 +     # 新鲜度权重30%
            consistency_score * 0.3      # 一致性权重30%
        )
        
        return overall_score
    
    def generate_health_report(self, knowledge_base, test_queries):
        """生成完整的健康度报告"""
        print("正在检查知识库健康度...")
        
        # 1. 检查缺少的知识
        print("1. 检查缺少的知识...")
        missing_result = self.check_missing_knowledge(knowledge_base, test_queries)
        
        # 2. 检查过期的知识
        print("2. 检查过期的知识...")
        outdated_result = self.check_outdated_knowledge(knowledge_base)
        
        # 3. 检查冲突的知识
        print("3. 检查冲突的知识...")
        conflicting_result = self.check_conflicting_knowledge(knowledge_base)
        
        # 4. 计算整体健康度
        overall_score = self.calculate_overall_health_score(missing_result, outdated_result, conflicting_result)
        
        # 5. 生成报告
        report = {
            "overall_health_score": overall_score,
            "health_level": self.get_health_level(overall_score),
            "missing_knowledge": missing_result,
            "outdated_knowledge": outdated_result,
            "conflicting_knowledge": conflicting_result,
            "recommendations": self.generate_recommendations(missing_result, outdated_result, conflicting_result),
            "check_date": datetime.now().isoformat()
        }
        
        return report
    
    def get_health_level(self, score):
        """根据评分确定健康等级"""
        if score >= 0.8:
            return "优秀"
        elif score >= 0.6:
            return "良好"
        elif score >= 0.4:
            return "一般"
        else:
            return "需要改进"
    
    def generate_recommendations(self, missing_result, outdated_result, conflicting_result):
        """生成改进建议"""
        recommendations = []
        
        # 基于缺少知识的建议
        missing_count = len(missing_result.get('missing_knowledge', []))
        if missing_count > 0:
            recommendations.append(f"补充{missing_count}个缺少的知识点，提高覆盖率")
        
        # 基于过期知识的建议
        outdated_count = len(outdated_result.get('outdated_knowledge', []))
        if outdated_count > 0:
            recommendations.append(f"更新{outdated_count}个过期知识点，确保信息时效性")
        
        # 基于冲突知识的建议
        conflicting_count = len(conflicting_result.get('conflicting_knowledge', []))
        if conflicting_count > 0:
            recommendations.append(f"解决{conflicting_count}个知识冲突，提高一致性")
        
        if not recommendations:
            recommendations.append("知识库状态良好，建议定期维护")
        
        return recommendations

def main():
    # 初始化知识库健康度检查器
    checker = KnowledgeBaseHealthChecker()
    
    print("=== 知识库健康度检查示例（迪士尼主题乐园） ===\n")
    
    # 示例知识库（包含一些故意的问题）
    knowledge_base = [
        {
            "id": "kb_001",
            "content": "上海迪士尼乐园位于上海市浦东新区，是中国大陆首座迪士尼主题乐园，于2016年6月16日开园。乐园占地面积390公顷，包含七大主题园区。",
            "last_updated": "2024-01-15"
        },
        {
            "id": "kb_002",
            "content": "上海迪士尼乐园的门票价格：平日成人票价为399元，周末和节假日为499元。儿童票平日为299元，周末为374元。",
            "last_updated": "2023-12-01"  # 故意设置为较旧的时间
        },
        {
            "id": "kb_003",
            "content": "上海迪士尼乐园门票价格：成人票平日350元，周末450元。儿童票平日250元，周末350元。",  # 故意设置冲突的价格
            "last_updated": "2024-02-01"
        },
        {
            "id": "kb_004",
            "content": "上海迪士尼乐园营业时间为上午8:00至晚上8:00，全年无休。",
            "last_updated": "2024-01-20"
        },
        {
            "id": "kb_005",
            "content": "从上海市区到迪士尼乐园可以乘坐地铁11号线到迪士尼站，或乘坐迪士尼专线巴士。",
            "last_updated": "2024-01-10"
        }
    ]
    
    # 测试查询
    test_queries = [
        {
            "query": "上海迪士尼乐园在哪里？",
            "expected_answer": "浦东新区"
        },
        {
            "query": "门票多少钱？",
            "expected_answer": "价格信息"
        },
        {
            "query": "营业时间是什么？",
            "expected_answer": "8:00-20:00"
        },
        {
            "query": "怎么去迪士尼？",
            "expected_answer": "地铁11号线"
        },
        {
            "query": "有什么特别活动？",  # 知识库中没有相关信息
            "expected_answer": "活动信息"
        },
        {
            "query": "停车费是多少？",  # 知识库中没有相关信息
            "expected_answer": "停车费信息"
        }
    ]
    
    # 生成健康度报告
    health_report = checker.generate_health_report(knowledge_base, test_queries)
    
    # 显示报告
    print("=== 知识库健康度报告 ===\n")
    
    print(f"整体健康度评分: {health_report['overall_health_score']:.2f}")
    print(f"健康等级: {health_report['health_level']}")
    print(f"检查时间: {health_report['check_date']}")
    
    print("\n" + "="*60 + "\n")
    
    # 详细分析
    print("=== 详细分析 ===\n")
    
    # 1. 缺少的知识
    print("1. 缺少的知识分析:")
    missing = health_report['missing_knowledge']
    print(f"   覆盖率: {health_report['missing_knowledge']['coverage_score']*100:.1f}%")
    print(f"   缺少知识点数量: {len(missing['missing_knowledge'])}")
    for i, item in enumerate(missing['missing_knowledge'][:3], 1):
        print(f"   {i}. 查询: {item['query']}")
        print(f"      缺少方面: {item['missing_aspect']}")
        print(f"      重要性: {item['importance']}")
    
    print("\n" + "-"*40 + "\n")
    
    # 2. 过期的知识
    print("2. 过期的知识分析:")
    outdated = health_report['outdated_knowledge']
    print(f"   新鲜度评分: {outdated['freshness_score']:.2f}")
    print(f"   过期知识点数量: {len(outdated['outdated_knowledge'])}")
    for i, item in enumerate(outdated['outdated_knowledge'][:3], 1):
        print(f"   {i}. 切片ID: {item['chunk_id']}")
        print(f"      过期方面: {item['outdated_aspect']}")
        print(f"      严重程度: {item['severity']}")
    
    print("\n" + "-"*40 + "\n")
    
    # 3. 冲突的知识
    print("3. 冲突的知识分析:")
    conflicting = health_report['conflicting_knowledge']
    print(f"   一致性评分: {conflicting['consistency_score']:.2f}")
    print(f"   冲突数量: {len(conflicting['conflicting_knowledge'])}")
    for i, item in enumerate(conflicting['conflicting_knowledge'][:3], 1):
        print(f"   {i}. 冲突类型: {item['conflict_type']}")
        print(f"      相关切片: {item['chunk_ids']}")
        print(f"      严重程度: {item['severity']}")
    
    print("\n" + "="*60 + "\n")
    
    # 改进建议
    print("=== 改进建议 ===\n")
    for i, recommendation in enumerate(health_report['recommendations'], 1):
        print(f"{i}. {recommendation}")

if __name__ == "__main__":
    main() 