"""基于 Assistant 实现的旅游智能助手

这个模块提供了一个智能地图助手，可以：
1、查询火车票
2、查询地图
3、规划路线
4、推荐景点
5. 支持多种交互方式（GUI、TUI、测试模式）
6. 支持旅游规划、地点查询、路线导航等功能
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
    """初始化旅游助手服务
    
    配置说明：
    - 使用 qwen-max 作为底层语言模型
    - 设置系统角色为旅游助手
    - 配置高德地图 MCP 工具
    - 配置12306 MCP 工具

    
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
    system = ('你扮演一个旅游助手，你具有查询火车票、查询地图、规划路线、推荐景点等能力。'
             '你可以帮助用户规划旅游行程，查找地点，导航等。'
             '你应该充分利用高德地图的各种功能来提供专业的建议。'
             '你应该充分利用12306的各种功能来提供专业的建议。')
    # MCP 工具配置
    tools = [{
        "mcpServers": {
            "12306-mcp": {
               "type": "sse",
               "url": "https://mcp.api-inference.modelscope.net/2b4e2b2e55a34a/sse"
            }
        }
    },{
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
            }
        }
    }]
    
    try:
        # 创建助手实例
        bot = Assistant(
            llm=llm_cfg,
            name='旅游助手',
            description='火车票查询、地图查询、路线规划、景点推荐',
            system_message=system,
            function_list=tools
        )
        print("助手初始化成功！")
        return bot
    except Exception as e:
        print(f"助手初始化失败: {str(e)}")
        raise

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
                '帮我规划从北京到上海的一日游行程，主要想去外滩和迪士尼',
                '我在南京路步行街，帮我找一家北京评分高的本帮菜餐厅',
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
    app_gui()          # 图形界面模式（默认）