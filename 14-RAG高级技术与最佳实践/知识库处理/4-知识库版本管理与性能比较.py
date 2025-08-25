# 知识库版本管理与性能比较
"""
如何对知识库进行版本管理，实现回归测试、上线前验收，并比较不同版本的知识库性能，选择最优版本。
TO DO：知识库版本管理与性能比较
    核心功能：
    • 版本创建：为知识库创建带描述和统计信息的版本
    • 哈希值计算：使用MD5计算版本的唯一标识
    • 统计信息：记录知识切片数量、内容长度、分类分布等
    • 版本比较：比较两个版本的差异和变化
"""
# 导入依赖库
import dashscope
import os
import json
import re
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import faiss
from openai import OpenAI

# 从环境变量中获取 API Key
os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here'
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

# 初始化百炼兼容的 OpenAI 客户端
client = OpenAI(
    api_key=dashscope.api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 全局配置
TEXT_EMBEDDING_MODEL = "text-embedding-v4"
TEXT_EMBEDDING_DIM = 1024

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

def get_text_embedding(text):
    """获取文本的 Embedding"""
    response = client.embeddings.create(
        model=TEXT_EMBEDDING_MODEL,
        input=text,
        dimensions=TEXT_EMBEDDING_DIM
    )
    return response.data[0].embedding

class KnowledgeBaseVersionManager:
    def __init__(self, model="qwen-turbo-latest"):
        self.model = model
        self.versions = {}
        
    def create_version(self, knowledge_base, version_name, description=""):
        """创建知识库版本"""
        # 构建向量索引
        metadata_store, text_index = self.build_vector_index(knowledge_base)
        
        version_info = {
            "version_name": version_name,
            "description": description,
            "created_date": datetime.now().isoformat(),
            "knowledge_base": knowledge_base,
            "metadata_store": metadata_store,
            "text_index": text_index,
            "statistics": self.calculate_version_statistics(knowledge_base)
        }
        
        self.versions[version_name] = version_info
        return version_info
    
    def build_vector_index(self, knowledge_base):
        """构建向量索引"""
        metadata_store = []
        text_vectors = []
        
        for i, chunk in enumerate(knowledge_base):
            content = chunk.get('content', '')
            if not content.strip():
                continue
                
            metadata = {
                "id": i,
                "content": content,
                "chunk_id": chunk.get('id', f'chunk_{i}')
            }
            
            # 获取文本embedding
            vector = get_text_embedding(content)
            text_vectors.append(vector)
            metadata_store.append(metadata)
        
        # 创建FAISS索引
        text_index = faiss.IndexFlatL2(TEXT_EMBEDDING_DIM)
        text_index_map = faiss.IndexIDMap(text_index)
        
        if text_vectors:
            text_ids = [m["id"] for m in metadata_store]
            text_index_map.add_with_ids(np.array(text_vectors).astype('float32'), np.array(text_ids))
        
        return metadata_store, text_index_map
    
    def calculate_version_statistics(self, knowledge_base):
        """计算版本统计信息"""
        total_chunks = len(knowledge_base)
        total_content_length = sum(len(chunk.get('content', '')) for chunk in knowledge_base)
        
        return {
            "total_chunks": total_chunks,
            "total_content_length": total_content_length,
            "average_chunk_length": total_content_length / total_chunks if total_chunks > 0 else 0
        }
    
    def compare_versions(self, version1_name, version2_name):
        """比较两个版本的差异"""
        if version1_name not in self.versions or version2_name not in self.versions:
            return {"error": "版本不存在"}
        
        v1 = self.versions[version1_name]
        v2 = self.versions[version2_name]
        
        kb1 = v1['knowledge_base']
        kb2 = v2['knowledge_base']
        
        comparison = {
            "version1": version1_name,
            "version2": version2_name,
            "comparison_date": datetime.now().isoformat(),
            "changes": self.detect_changes(kb1, kb2),
            "statistics_comparison": self.compare_statistics(v1['statistics'], v2['statistics'])
        }
        
        return comparison
    
    def detect_changes(self, kb1, kb2):
        """检测知识库变化"""
        changes = {
            "added_chunks": [],
            "removed_chunks": [],
            "modified_chunks": [],
            "unchanged_chunks": []
        }
        
        # 创建ID映射
        kb1_dict = {chunk.get('id'): chunk for chunk in kb1}
        kb2_dict = {chunk.get('id'): chunk for chunk in kb2}
        
        # 检测新增和删除
        kb1_ids = set(kb1_dict.keys())
        kb2_ids = set(kb2_dict.keys())
        
        added_ids = kb2_ids - kb1_ids
        removed_ids = kb1_ids - kb2_ids
        common_ids = kb1_ids & kb2_ids
        
        # 记录新增的知识切片
        for chunk_id in added_ids:
            changes["added_chunks"].append({
                "id": chunk_id,
                "content": kb2_dict[chunk_id].get('content', '')
            })
        
        # 记录删除的知识切片
        for chunk_id in removed_ids:
            changes["removed_chunks"].append({
                "id": chunk_id,
                "content": kb1_dict[chunk_id].get('content', '')
            })
        
        # 检测修改的知识切片
        for chunk_id in common_ids:
            chunk1 = kb1_dict[chunk_id]
            chunk2 = kb2_dict[chunk_id]
            
            if chunk1.get('content') != chunk2.get('content'):
                changes["modified_chunks"].append({
                    "id": chunk_id,
                    "old_content": chunk1.get('content', ''),
                    "new_content": chunk2.get('content', '')
                })
            else:
                changes["unchanged_chunks"].append(chunk_id)
        
        return changes
    
    def compare_statistics(self, stats1, stats2):
        """比较统计信息"""
        comparison = {}
        
        for key in stats1.keys():
            if key in stats2:
                if isinstance(stats1[key], (int, float)):
                    comparison[key] = {
                        "version1": stats1[key],
                        "version2": stats2[key],
                        "difference": stats2[key] - stats1[key],
                        "percentage_change": ((stats2[key] - stats1[key]) / stats1[key] * 100) if stats1[key] != 0 else 0
                    }
                elif isinstance(stats1[key], dict):
                    comparison[key] = self.compare_dict_statistics(stats1[key], stats2[key])
        
        return comparison
    
    def compare_dict_statistics(self, dict1, dict2):
        """比较字典类型的统计信息"""
        comparison = {}
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        for key in all_keys:
            val1 = dict1.get(key, 0)
            val2 = dict2.get(key, 0)
            comparison[key] = {
                "version1": val1,
                "version2": val2,
                "difference": val2 - val1
            }
        
        return comparison
    
    def evaluate_version_performance(self, version_name, test_queries):
        """评估版本性能"""
        if version_name not in self.versions:
            return {"error": "版本不存在"}
        
        performance_metrics = {
            "version_name": version_name,
            "evaluation_date": datetime.now().isoformat(),
            "query_results": [],
            "overall_metrics": {}
        }
        
        total_queries = len(test_queries)
        correct_answers = 0
        response_times = []
        
        for query_info in test_queries:
            query = query_info['query']
            expected_answer = query_info.get('expected_answer', '')
            
            # 使用embedding检索
            start_time = datetime.now()
            retrieved_chunks = self.retrieve_relevant_chunks(query, version_name)
            end_time = datetime.now()
            
            response_time = (end_time - start_time).total_seconds()
            response_times.append(response_time)
            
            # 评估检索质量
            is_correct = self.evaluate_retrieval_quality(query, retrieved_chunks, expected_answer)
            if is_correct:
                correct_answers += 1
            
            performance_metrics["query_results"].append({
                "query": query,
                "retrieved_chunks": len(retrieved_chunks),
                "response_time": response_time,
                "is_correct": is_correct
            })
        
        # 计算整体指标
        accuracy = correct_answers / total_queries if total_queries > 0 else 0
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        performance_metrics["overall_metrics"] = {
            "accuracy": accuracy,
            "avg_response_time": avg_response_time,
            "total_queries": total_queries,
            "correct_answers": correct_answers
        }
        
        return performance_metrics
    
    def retrieve_relevant_chunks(self, query, version_name, k=3):
        """使用embedding和faiss检索相关知识切片"""
        if version_name not in self.versions:
            return []
        
        version_info = self.versions[version_name]
        metadata_store = version_info['metadata_store']
        text_index = version_info['text_index']
        
        # 获取查询的embedding
        query_vector = np.array([get_text_embedding(query)]).astype('float32')
        
        # 使用faiss进行检索
        distances, indices = text_index.search(query_vector, k)
        
        relevant_chunks = []
        for i, doc_id in enumerate(indices[0]):
            if doc_id != -1:  # faiss返回-1表示没有找到匹配
                # 通过ID在元数据中查找
                match = next((item for item in metadata_store if item["id"] == doc_id), None)
                if match:
                    # 构造返回的知识切片格式
                    chunk = {
                        "id": match["chunk_id"],
                        "content": match["content"],
                        "similarity_score": 1.0 / (1.0 + distances[0][i])  # 将距离转换为相似度
                    }
                    relevant_chunks.append(chunk)
        
        return relevant_chunks
    
    def evaluate_retrieval_quality(self, query, retrieved_chunks, expected_answer):
        """评估检索质量"""
        if not retrieved_chunks:
            return False
        
        # 简化的质量评估
        for chunk in retrieved_chunks:
            # 获取知识切片内容并转换为小写,如果content不存在则返回空字符串
            # 使用get方法安全获取content字段,避免KeyError异常
            # lower()方法将文本转换为小写,便于后续不区分大小写的匹配
            content = chunk.get('content', '').lower()
            if expected_answer.lower() in content:
                return True
        
        return False
    
    def compare_version_performance(self, version1_name, version2_name, test_queries):
        """比较两个版本的性能"""
        perf1 = self.evaluate_version_performance(version1_name, test_queries)
        perf2 = self.evaluate_version_performance(version2_name, test_queries)
        
        if "error" in perf1 or "error" in perf2:
            return {"error": "版本评估失败"}
        
        comparison = {
            "version1": version1_name,
            "version2": version2_name,
            "comparison_date": datetime.now().isoformat(),
            "performance_comparison": {
                "accuracy": {
                    "version1": perf1["overall_metrics"]["accuracy"],
                    "version2": perf2["overall_metrics"]["accuracy"],
                    "improvement": perf2["overall_metrics"]["accuracy"] - perf1["overall_metrics"]["accuracy"]
                },
                "response_time": {
                    "version1": perf1["overall_metrics"]["avg_response_time"],
                    "version2": perf2["overall_metrics"]["avg_response_time"],
                    "improvement": perf1["overall_metrics"]["avg_response_time"] - perf2["overall_metrics"]["avg_response_time"]
                }
            },
            "recommendation": self.generate_performance_recommendation(perf1, perf2)
        }
        
        return comparison
    
    def generate_performance_recommendation(self, perf1, perf2):
        """生成性能建议"""
        acc1 = perf1["overall_metrics"]["accuracy"]
        acc2 = perf2["overall_metrics"]["accuracy"]
        time1 = perf1["overall_metrics"]["avg_response_time"]
        time2 = perf2["overall_metrics"]["avg_response_time"]
        
        if acc2 > acc1 and time2 <= time1:
            return f"推荐使用版本2，准确率提升{(acc2-acc1)*100:.1f}%，响应时间{'提升' if time2 < time1 else '相当'}"
        elif acc2 > acc1 and time2 > time1:
            return f"版本2准确率更高但响应时间较长，需要权衡"
        elif acc2 < acc1 and time2 < time1:
            return f"版本2响应更快但准确率较低，需要权衡"
        else:
            return f"推荐使用版本1，性能更优"
    
    def generate_regression_test(self, version_name, test_queries):
        """生成回归测试"""
        if version_name not in self.versions:
            return {"error": "版本不存在"}
        
        regression_results = {
            "version_name": version_name,
            "test_date": datetime.now().isoformat(),
            "test_results": [],
            "pass_rate": 0
        }
        
        passed_tests = 0
        total_tests = len(test_queries)
        
        for query_info in test_queries:
            query = query_info['query']
            expected_answer = query_info.get('expected_answer', '')
            
            # 执行测试
            retrieved_chunks = self.retrieve_relevant_chunks(query, version_name)
            is_passed = self.evaluate_retrieval_quality(query, retrieved_chunks, expected_answer)
            
            if is_passed:
                passed_tests += 1
            
            regression_results["test_results"].append({
                "query": query,
                "expected": expected_answer,
                "retrieved": len(retrieved_chunks),
                "passed": is_passed
            })
        
        regression_results["pass_rate"] = passed_tests / total_tests if total_tests > 0 else 0
        
        return regression_results

def main():
    # 初始化版本管理器
    version_manager = KnowledgeBaseVersionManager()
    
    print("=== 知识库版本管理与性能比较示例（迪士尼主题乐园） ===\n")
    
    # 创建版本1（基础版本）
    knowledge_base_v1 = [
        {
            "id": "kb_001",
            "content": "上海迪士尼乐园位于上海市浦东新区，是中国大陆首座迪士尼主题乐园，于2016年6月16日开园。"
        },
        {
            "id": "kb_002",
            "content": "上海迪士尼乐园的门票价格：平日成人票价为399元，周末和节假日为499元。"
        },
        {
            "id": "kb_003",
            "content": "上海迪士尼乐园营业时间为上午8:00至晚上8:00。"
        }
    ]
    
    # 创建版本2（增强版本）
    knowledge_base_v2 = [
        {
            "id": "kb_001",
            "content": "上海迪士尼乐园位于上海市浦东新区，是中国大陆首座迪士尼主题乐园，于2016年6月16日开园。乐园占地面积390公顷，包含七大主题园区。"
        },
        {
            "id": "kb_002",
            "content": "上海迪士尼乐园的门票价格：平日成人票价为399元，周末和节假日为499元。儿童票（1.0-1.4米）平日为299元，周末为374元。1.0米以下儿童免费。"
        },
        {
            "id": "kb_003",
            "content": "上海迪士尼乐园营业时间为上午8:00至晚上8:00，全年无休。建议出发前查看官方网站确认具体时间。"
        },
        {
            "id": "kb_004",
            "content": "从上海市区到迪士尼乐园可以乘坐地铁11号线到迪士尼站，或乘坐迪士尼专线巴士。"
        },
        {
            "id": "kb_005",
            "content": "上海迪士尼乐园的特色项目包括：创极速光轮、七个小矮人矿山车、加勒比海盗等。"
        }
    ]
    
    # 功能1: 创建版本
    print("功能1: 创建知识库版本")
    v1_info = version_manager.create_version(knowledge_base_v1, "v1.0", "基础版本")
    v2_info = version_manager.create_version(knowledge_base_v2, "v2.0", "增强版本")
    
    print(f"版本1信息:")
    print(f"  版本名: {v1_info['version_name']}")
    print(f"  描述: {v1_info['description']}")
    print(f"  知识切片数量: {v1_info['statistics']['total_chunks']}")
    print(f"  平均切片长度: {v1_info['statistics']['average_chunk_length']:.0f}字符")
    
    print(f"\n版本2信息:")
    print(f"  版本名: {v2_info['version_name']}")
    print(f"  描述: {v2_info['description']}")
    print(f"  知识切片数量: {v2_info['statistics']['total_chunks']}")
    print(f"  平均切片长度: {v2_info['statistics']['average_chunk_length']:.0f}字符")
    
    print("\n" + "="*60 + "\n")
    
    # 功能示例2: 版本比较
    print("功能2: 版本差异比较")
    comparison = version_manager.compare_versions("v1.0", "v2.0")
    
    print(f"版本比较结果:")
    changes = comparison['changes']
    print(f"  新增知识切片: {len(changes['added_chunks'])}个")
    print(f"  删除知识切片: {len(changes['removed_chunks'])}个")
    print(f"  修改知识切片: {len(changes['modified_chunks'])}个")
    
    print(f"\n新增的知识切片:")
    for i, chunk in enumerate(changes['added_chunks'], 1):
        print(f"  {i}. ID: {chunk['id']}")
        print(f"     内容: {chunk['content']}")
    
    print(f"\n修改的知识切片:")
    for i, chunk in enumerate(changes['modified_chunks'], 1):
        print(f"  {i}. ID: {chunk['id']}")
        print(f"     旧内容: {chunk['old_content']}")
        print(f"     新内容: {chunk['new_content']}")
    
    print("\n" + "="*60 + "\n")
    
    # 功能3: 性能评估
    print("功能3: 版本性能评估")
    
    test_queries = [
        {"query": "上海迪士尼乐园在哪里？", "expected_answer": "浦东新区"}, # 关键词包含即正确
        {"query": "门票多少钱？", "expected_answer": "价格"},
        {"query": "营业时间是什么？", "expected_answer": "8:00"},
        {"query": "怎么去迪士尼？", "expected_answer": "地铁"},
        {"query": "有什么好玩的项目？", "expected_answer": "项目"}
    ]
    
    perf_v1 = version_manager.evaluate_version_performance("v1.0", test_queries)
    perf_v2 = version_manager.evaluate_version_performance("v2.0", test_queries)
    
    print(f"版本1性能:")
    print(f"  准确率: {perf_v1['overall_metrics']['accuracy']*100:.1f}%")
    print(f"  平均响应时间: {perf_v1['overall_metrics']['avg_response_time']*1000:.1f}ms")
    
    print(f"\n版本2性能:")
    print(f"  准确率: {perf_v2['overall_metrics']['accuracy']*100:.1f}%")
    print(f"  平均响应时间: {perf_v2['overall_metrics']['avg_response_time']*1000:.1f}ms")
    
    print("\n" + "="*60 + "\n")
    
    # 功能4: 性能比较
    print("功能4: 性能比较与建议")
    perf_comparison = version_manager.compare_version_performance("v1.0", "v2.0", test_queries)
    
    print(f"性能比较结果:")
    comp = perf_comparison['performance_comparison']
    print(f"  准确率提升: {comp['accuracy']['improvement']*100:.1f}%")
    print(f"  响应时间变化: {comp['response_time']['improvement']*1000:.1f}ms")
    print(f"  建议: {perf_comparison['recommendation']}")
    
    print("\n" + "="*60 + "\n")
    
    # 功能5: 回归测试
    print("功能5: 回归测试")
    regression_v2 = version_manager.generate_regression_test("v2.0", test_queries)
    
    print(f"回归测试结果:")
    print(f"  测试通过率: {regression_v2['pass_rate']*100:.1f}%")
    print(f"  测试用例数量: {len(regression_v2['test_results'])}")
    
    print(f"\n详细测试结果:")
    for i, result in enumerate(regression_v2['test_results'], 1):
        status = "✓" if result['passed'] else "✗"
        print(f"  {i}. {result['query']} {status}")

if __name__ == "__main__":
    main() 