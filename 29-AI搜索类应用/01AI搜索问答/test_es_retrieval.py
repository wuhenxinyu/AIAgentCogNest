import os
import json
import logging
from .tools.es_retrieval import ESRetrievalTool

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_retrieval():
    """独立测试 Elasticsearch 检索工具的功能。"""
    print("--- 开始独立测试 Elasticsearch 检索工具 ---")

    # --- 步骤 1: 准备输入参数 ---
    query = "介绍下雇主责任险"
    docs_folder = 'docs'
    files_to_process = [os.path.join(docs_folder, f) for f in os.listdir(docs_folder) if os.path.isfile(os.path.join(docs_folder, f))]

    print("\n--- 步骤 1: 准备输入参数 ---")
    print(f"查询语句: {query}")
    print(f"待处理文件数: {len(files_to_process)}")

    # 准备工具的配置
    # 确保 'es' 配置部分与您的 Elasticsearch 实例匹配
    tool_cfg = {
        'es': {
            'host': 'https://localhost', # 或者您的 ES 主机
            'port': 9200,
            'index_name': 'my_insurance_docs_index_test', # 使用专用的测试索引
            # 如果您的 Elasticsearch 开启了认证，请取消下面的注释并填写正确的用户名和密码
            'user': 'elastic',
            'password': 'euqPcOlHrmW18rtaS-3P',
        }
    }

    # 实例化工具
    retrieval_tool = ESRetrievalTool(cfg=tool_cfg)

    # 准备 call 方法的参数
    tool_params = {
        'query': query,
        'files': files_to_process
    }
    
    # --- 步骤 2: 调用工具的 call 方法 (此过程会进行索引和搜索) ---
    print("\n--- 步骤 2: 调用工具的 call 方法 (此过程会进行索引和搜索) ---")
    try:
        # 调用工具，它会返回一个 JSON 字符串
        results_str = retrieval_tool.call(params=tool_params)
        
        # 解析返回的 JSON 字符串
        results = json.loads(results_str)

    except Exception as e:
        print(f"\n--- 测试失败：调用 retrieval_tool.call 时发生严重错误: {e} ---")
        import traceback
        traceback.print_exc()
        print("\n--- 测试结束 ---")
        return

    # --- 步骤 3: 解析并展示检索结果 ---
    print("\n--- 步骤 3: 解析并展示检索结果 ---")
    if not results:
        print("检索结果为空。可能原因：")
        print("1. 'docs' 文件夹中的文档内容与查询不相关。")
        print("2. Elasticsearch 索引或搜索过程存在问题。")
        print("3. 请检查 ES 服务日志以获取更多线索。")
    else:
        print(f"成功从 Elasticsearch 检索到 {len(results)} 条相关内容：\n")
        for i, hit in enumerate(results):
            print(f"--- 结果 {i+1} ---")
            print(f"来源: {hit.get('url', '未知')}")
            # 注意：在当前版本的工具中，我们不再返回得分
            # print(f"得分: {hit.get('score', '未知')}") 
            # 截断内容以方便预览
            content_preview = hit.get('text', '')[:300]
            print(f"内容预览: {content_preview}...")
            print()

    print("\n--- 测试结束 ---")

if __name__ == '__main__':
    test_retrieval() 