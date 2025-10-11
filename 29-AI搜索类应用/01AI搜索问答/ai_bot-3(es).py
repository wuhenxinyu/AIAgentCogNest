"""
RAG使用Elasticsearch检索文档
    用 elasticsearch来进行文档片段chunks进行索引和管理，提升检索性能。帮我扩展 qwen_agent文件夹中的代码，新增该功能实现，放到 qwen_agent对应的文件夹下面即可。
    然后让 Assistant中的memory，有一个选项是可以用es来实现检索和管理文档片段chunks。
    实现了 es+bm25的 RAG策略

ES的相关度得分是如何计算的？
    这个得分是 Elasticsearch 的核心功能，使用的BM25算法（Best Match 25）可以将其理解为一个综合打分机制，主要考虑以下三个因素：词频（Term Frequency - TF）、逆文档频率（Inverse Document Frequency - IDF）和文档长度归一化（Document Length Normalization）。
    
"""
import os
import asyncio
from typing import Optional
from .agents import Assistant
from .gui import WebUI
import warnings
warnings.filterwarnings("ignore")
os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here'
def init_agent_service():
    """初始化具备 Elasticsearch RAG 能力的助手服务"""
    
    # 步骤 1: LLM 配置
    llm_cfg = {
        'model': 'qwen-max',
        'model_server': 'dashscope',
        'api_key': os.getenv('DASHSCOPE_API_KEY'),
        'generate_cfg': {
            'top_p': 0.8
        }
    }

    # 步骤 2: RAG 配置 - 激活并配置 Elasticsearch 后端
    rag_cfg = {
        "rag_backend": "elasticsearch",  # 关键：指定使用 ES 后端
        "es": {
            "host": "https://localhost",
            "port": 9200,
            "user": "elastic",
            "password": "euqPcOlHrmW18rtaS-3P",  # 您的 Elasticsearch 密码
            "index_name": "my_insurance_docs_index" # 自定义索引名称
        },
        "parser_page_size": 500 # 文档分块大小
    }

    # 步骤 3: 系统指令和工具
    system_instruction = '''你是一个基于本地知识库的AI助手。
请根据用户的问题，利用检索工具从知识库中查找最相关的信息，并结合这些信息给出专业、准确的回答。'''

    # 获取文件夹下所有文件
    file_dir = os.path.join(os.path.dirname(__file__), 'docs')
    files = []
    if os.path.exists(file_dir):
        for file in os.listdir(file_dir):
            file_path = os.path.join(file_dir, file)
            if os.path.isfile(file_path):
                files.append(file_path)
    print('知识库文件列表:', files)

    # 步骤 4: 创建智能体实例
    # 通过 rag_cfg 参数传入我们的 ES 配置
    bot = Assistant(
        llm=llm_cfg,
        system_message=system_instruction,
        files=files,
        rag_cfg=rag_cfg
    )
    return bot

def main():
    """启动 Web 图形界面"""
    try:
        print("正在启动 AI 助手 Web 界面 (Elasticsearch 后端)...")
        bot = init_agent_service()
        chatbot_config = {
            'prompt.suggestions': [
                '介绍下雇主责任险',
                '雇主责任险和工伤保险有什么主要区别？',
                '介绍一下平安商业综合责任保险（亚马逊）的保障范围。',
                '施工保主要适用于哪些场景？',
            ]
        }
        WebUI(bot, chatbot_config=chatbot_config).run()
    except Exception as e:
        print(f"启动 Web 界面失败: {e}")
        print("请检查网络连接、API Key 以及 Elasticsearch 服务是否正常运行。")

if __name__ == '__main__':
    main() 