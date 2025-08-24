"""基于 Assistant 实现的高德地图智能助手

这个模块提供了一个智能地图助手，可以：
1. 通过自然语言进行地图服务查询
2. 支持多种交互方式（GUI、TUI、测试模式）
3. 支持旅游规划、地点查询、路线导航等功能
"""

import os
import asyncio
from typing import Optional
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI

# # 配置 DashScope
# dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')  # 从环境变量获取 API Key
# dashscope.timeout = 30  # 设置超时时间为 30 秒

def init_agent_service():
    """初始化高德地图助手服务
    
    配置说明：
    - 使用 qwen-max 作为底层语言模型
    - 设置系统角色为地图助手
    - 配置高德地图 MCP 工具
    
    Returns:
        Assistant: 配置好的地图助手实例
    """
    # LLM 模型配置
    llm_cfg = {
        'model': 'qwen-turbo',
        'timeout': 30,  # 设置模型调用超时时间
        'retry_count': 3,  # 设置重试次数
        'model_server': 'https://api.fe8.cn/v1',  # 从环境变量获取自定义模型API基础地址
        'api_key': "sk-siRF78nIxVVBKekhvZF6POAzSrFXymXwzCFj4YT6SzFIlvWA",  # 从环境变量获取自定义模型API密钥
    }
    # 系统角色设定
    system = ('你扮演一个地图助手，你具有查询地图、规划路线、推荐景点等能力。'
             '你可以帮助用户规划旅游行程，查找地点，导航等。'
             '你应该充分利用高德地图的各种功能来提供专业的建议。')
    # MCP 工具配置
    tools = [{
        "mcpServers": {
            "amap-maps": {
                "type": "sse",
                "url": "https://mcp.api-inference.modelscope.net/bb3c9d1d31364c/sse"
            }
        }
    }]
    
    try:
        # 创建助手实例
        bot = Assistant(
            llm=llm_cfg,
            name='地图助手',
            description='地图查询与路线规划',
            system_message=system,
            function_list=tools,
        )
        print("助手初始化成功！")
        return bot
    except Exception as e:
        print(f"助手初始化失败: {str(e)}")
        raise


def test(query='帮我查找上海东方明珠的具体位置', file: Optional[str] = None):
    """测试模式
    
    用于快速测试单个查询
    
    Args:
        query: 查询语句，默认为查询地标位置
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
    - 智能路线规划
    """
    try:
        print("正在启动 Web 界面...")
        # 初始化助手
        bot = init_agent_service()
        # 配置聊天界面
        chatbot_config = {
            'prompt.suggestions': [
                '帮我规划上海一日游行程，主要想去外滩和迪士尼',
                '我在南京路步行街，帮我找一家评分高的本帮菜餐厅',
                '从浦东机场到外滩怎么走最方便？',
                '推荐上海三个适合拍照的网红景点',
                '帮我查找上海科技馆的具体地址和营业时间',
                '从徐家汇到外滩有哪些公交路线？',
                '现在在豫园，附近有什么好玩的地方推荐？',
                '帮我找一下静安寺附近的停车场',
                '上海野生动物园到迪士尼乐园怎么走？',
                '推荐陆家嘴附近的高档餐厅'
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