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

# 定义资源文件根目录
ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')

# 配置 DashScope
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
                "command": "npx",
                "args": [
                    "-y",
                    "@amap/amap-maps-mcp-server"
                ],
                "env": {
                    "AMAP_MAPS_API_KEY": "8f05da4a643d3fe38b8bd06a7ef14139"
                }
            },
            "fetch": {
                "type": "sse",
                "url": "https://mcp.api-inference.modelscope.net/13c541d5a6e44c/sse"
            },
            # "bing-cn-mcp-server": {
            #     "type": "sse",
            #     "url": "https://mcp.api-inference.modelscope.net/5b769804d39d4d/sse"
            # }
        }
    }]
    
    try:
        # 创建助手实例
        bot = Assistant(
            llm=llm_cfg,
            name='AI助手',
            description='地图查询/指定网页获取/Bing搜索',
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
                # 初始化一个空列表用于存储响应，但后续该变量会被覆盖，此初始化无实际作用
                # 此处原代码初始化 response 列表可能是多余的，因为后续会在循环中重新赋值
                response = []
                # 调用 bot 的 run 方法，该方法会返回一个可迭代对象，用于逐步获取助手的响应
                # 遍历这个可迭代对象，每次迭代都会得到一个新的响应片段
                for response in bot.run(messages):
                    # 打印当前获取到的助手响应片段
                    print('bot response:', response)
                
                # 将最后一次获取到的响应添加到对话历史记录 messages 中
                # 这样可以保存完整的对话上下文，以便后续继续对话
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
                '将 https://k.sina.com.cn/article_7732457677_1cce3f0cd01901eeeq.html 网页转化为Markdown格式',
                '帮我找一下静安寺附近的停车场',
                '推荐陆家嘴附近的高档餐厅',
                '帮我搜索一下关于AI的最新新闻'
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