"""基于 Assistant 实现的文本计数智能助手

这个模块提供了一个智能文本计数助手，可以：
    1. 通过自然语言进行文本分析
    2. 支持多种交互方式（GUI、TUI、测试模式）
    3. 支持文本字符数、单词数、行数统计等功能

调用server中python文件本地sdk启动方式: a、python启动  b、uv启动运行脚本（在MCP协议中，uv工具常用于启动和管理MCP服务器。例如，在配置MCP服务器时，可以通过uv命令指定运行的Python脚本。）
"""

import os
import asyncio
from typing import Optional
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI

# 配置 DashScope
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', 'your_api_key_here')  # 从环境变量获取 API Key
dashscope.timeout = 30  # 设置超时时间为 30 秒

def init_agent_service():
    """初始化文本计数助手服务
    
    配置说明：
    - 使用 qwen-max 作为底层语言模型
    - 设置系统角色为文本分析助手
    - 配置文本计数 MCP 工具
    
    Returns:
        Assistant: 配置好的文本计数助手实例
    """
    # LLM 模型配置
    llm_cfg = {
        'model': 'qwen-max',
        'timeout': 30,  # 设置模型调用超时时间
        'retry_count': 3,  # 设置重试次数
    }
    # 系统角色设定
    system = ('你扮演一个文本分析助手，你具有计算文本字符数、单词数、行数等能力。'
             '你可以帮助用户分析文本的基本统计信息。'
             '你应该充分利用文本计数服务的功能来提供专业的分析。')
    # MCP 工具配置
    tools = [{
        "mcpServers": {
            # 方式一：python 启动 mcp server
            # "txt-counter": {
            #     "command": "python",
            #     "args": ["18-MCP与A2A的应用/MCP应用/server/txt_counter.py"],
            #     "port": 6277
            # }
            
            # 方式二：uv被用来运行txt-counter.py脚本，从而启动MCP服务器
            "txt-counter": {
                "name": "txt-counter",
                "type": "stdio",
                "command": "uv",
                "args": [
                    "--directory",
                    "/Users/clz/Downloads/xmkf/llm/AIAgentCogNest/18-MCP与A2A的应用/MCP应用/server",
                    "run",
                    "txt_counter.py"
                ]
           }
        }
    }]
    
    try:
        # 创建助手实例
        bot = Assistant(
            llm=llm_cfg,
            name='文本计数助手',
            description='文本统计与分析',
            system_message=system,
            function_list=tools,
        )
        print("助手初始化成功！")
        return bot
    except Exception as e:
        print(f"助手初始化失败: {str(e)}")
        raise

def test(query='帮我统计这段文本的字数：Hello World', file: Optional[str] = None):
    """测试模式
    
    用于快速测试单个查询
    
    Args:
        query: 查询语句，默认为简单的文本统计
        file: 可选的输入文件路径
    """
    try:
        # 初始化助手
        bot = init_agent_service()

        # 构建对话消息
        messages = []

        # 根据是否有文件输入构建不同的消息格式
        if not file:
            messages.append({'role': 'user', 'content': query})
        else:
            messages.append({'role': 'user', 'content': [{'text': query}, {'file': file}]})

        print("正在处理您的请求...")
        # 运行助手并打印响应
        for response in bot.run(messages):
            print('bot response:', response)
    except Exception as e:
        print(f"处理请求时出错: {str(e)}")

def app_tui():
    """终端交互模式
    
    提供命令行交互界面，支持：
    - 连续对话
    - 文件输入
    - 实时响应
    """
    try:
        # 初始化助手
        bot = init_agent_service()

        # 对话历史
        messages = []
        while True:
            try:
                # 获取用户输入
                query = input('user question: ')
                # 获取可选的文件输入
                file = input('file url (press enter if no file): ').strip()
                
                # 输入验证
                if not query:
                    print('user question cannot be empty！')
                    continue
                    
                # 构建消息
                if not file:
                    messages.append({'role': 'user', 'content': query})
                else:
                    messages.append({'role': 'user', 'content': [{'text': query}, {'file': file}]})

                print("正在处理您的请求...")
                # 运行助手并处理响应
                response = []
                for response in bot.run(messages):
                    print('bot response:', response)
                messages.extend(response)
            except Exception as e:
                print(f"处理请求时出错: {str(e)}")
                print("请重试或输入新的问题")
    except Exception as e:
        print(f"启动终端模式失败: {str(e)}")

def app_gui():
    """图形界面模式
    
    提供 Web 图形界面，特点：
    - 友好的用户界面
    - 预设查询建议
    - 智能文本分析
    """
    try:
        print("正在启动 Web 界面...")
        # 初始化助手
        bot = init_agent_service()
        # 配置聊天界面
        chatbot_config = {
            'prompt.suggestions': [
                '帮我统计这段文本的字数：Hello World',
                '计算下面这段文本的行数：\n第一行\n第二行\n第三行',
                '这段文本有多少个单词：The quick brown fox jumps over the lazy dog',
                '分析这段多语言文本：你好，世界！Hello World!',
                '统计下面文本的标点符号数量：Hello, World! How are you?',
                '帮我计算这段代码的行数：\ndef hello():\n    print("Hello")\n    return True',
                '这段文本中有多少个中文字符：你好，这是一段中文文本',
                '统计这段文本的空格数量：Hello  World   !',
                '计算这段文本的总字符数（包括空格和标点）',
                '分析这段文本的基本信息（字数、行数、单词数）'
            ]
        }
        
        print("Web 界面准备就绪，正在启动服务...")
        # 启动 Web 界面
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
    except Exception as e:
        print(f"启动 Web 界面失败: {str(e)}")
        print("请检查网络连接和 API Key 配置")

if __name__ == '__main__':
    # 运行模式选择
    # test()           # 测试模式
    # app_tui()        # 终端交互模式
    app_gui()          # 图形界面模式（默认） 