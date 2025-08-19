import os
import requests
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI

from qwen_agent.tools.base import BaseTool, register_tool

# 配置 DashScope
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', 'sk-58f051ae745e4bb19fdca31735105b11')  # 从环境变量获取 API Key
dashscope.timeout = 30  # 设置超时时间为 30 秒

functions_desc = [
    {
        "name": "get_current_weather",
        "description": "获取指定位置的当前天气情况",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市名称，例如：北京",
                },
                "adcode": {
                    "type": "string",
                    "description": "城市编码，例如：110000（北京）",
                }
            },
            "required": ["location"],
        },
    }
]

# ====== 天气查询工具实现 ======
@register_tool('get_current_weather')
class WeatherTool(BaseTool):
    """
    天气查询工具，通过高德地图API查询指定位置的天气情况。
    """
    description = '获取指定位置的当前天气情况'
    # 定义 WeatherTool 工具所需的参数列表
    # 每个参数是一个字典，包含参数名称、类型、描述和是否为必填项等信息
    parameters = [
        {
            # 参数名称为 location，用于指定要查询天气的城市名称
            'name': 'location',
            # 参数类型为字符串
            'type': 'string',
            # 参数描述，说明该参数是城市名称，并给出示例
            'description': '城市名称，例如：北京',
            # 该参数为必填项
            'required': True
        },
        {
            # 参数名称为 adcode，用于指定要查询天气的城市编码
            'name': 'adcode',
            # 参数类型为字符串
            'type': 'string',
            # 参数描述，说明该参数是城市编码，并给出北京的示例
            'description': '城市编码，例如：110000（北京）',
            # 该参数为非必填项
            'required': False
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        import json
        args = json.loads(params)
        location = args['location']
        adcode = args.get('adcode', None)
        
        return self.get_weather_from_gaode(location, adcode)
    
    def get_weather_from_gaode(self, location: str, adcode: str = None) -> str:
        """调用高德地图API查询天气"""
        gaode_api_key = "你的API KEY"  # 高德API Key
        base_url = "https://restapi.amap.com/v3/weather/weatherInfo"
        
        params = {
            "key": gaode_api_key,
            "city": adcode if adcode else location,
            "extensions": "base",  # 可改为 "all" 获取预报
        }
        
        try:
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == '1' and data.get('lives'):
                    weather_info = data['lives'][0]
                    result = f"天气查询结果：\n城市：{weather_info.get('city')}\n天气：{weather_info.get('weather')}\n温度：{weather_info.get('temperature')}°C\n风向：{weather_info.get('winddirection')}\n风力：{weather_info.get('windpower')}\n湿度：{weather_info.get('humidity')}%\n发布时间：{weather_info.get('reporttime')}"
                    return result
                else:
                    return f"获取天气信息失败：{data.get('info', '未知错误')}"
            else:
                return f"请求失败：HTTP状态码 {response.status_code}"
        except Exception as e:
            return f"获取天气信息出错：{str(e)}"

# ====== 初始化助手服务 ======
def init_agent_service():
    """初始化助手服务"""
    llm_cfg = {
        'model': 'qwen-turbo-2025-04-28',
        'timeout': 30,
        'retry_count': 3,
    }
    try:
        # 实例化一个 Assistant 对象，该对象代表一个 AI 助手
        # Assistant 是由 qwen_agent.agents 模块提供的类，用于创建具备特定功能的 AI 助手
        bot = Assistant(
            # llm 参数用于配置大语言模型的相关信息
            # llm_cfg 是一个字典，包含模型名称、超时时间和重试次数等配置
            # 这里使用的模型是 'qwen-turbo-2025-04-28'，超时时间为 30 秒，重试次数为 3 次
            llm=llm_cfg,
            # name 参数用于设置助手的名称
            # 这里将助手命名为 '天气助手'，方便在交互过程中识别
            name='天气助手',
            # description 参数用于简要描述助手的功能
            # 表明该助手的主要功能是查询天气
            description='天气助手，查询天气',
            # system_message 参数用于设置系统消息
            # 系统消息会作为助手的初始指令，告知助手其角色定位
            # 这里设置为 "你是一名有用的助手"，提示助手要为用户提供有用的帮助
            system_message="你是一名有用的助手",
            # function_list 参数用于指定助手可以调用的工具列表
            # 这里添加了 'get_current_weather' 工具，使助手能够调用天气查询功能
            function_list=['get_current_weather'],  # 增加天气工具
        )
        print("助手初始化成功！")
        return bot
    except Exception as e:
        print(f"助手初始化失败: {str(e)}")
        raise

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
    """图形界面模式，提供 Web 图形界面"""
    try:
        print("正在启动 Web 界面...")
        # 初始化助手
        bot = init_agent_service()
        # 配置聊天界面，列举3个典型门票查询问题
        chatbot_config = {
            'prompt.suggestions': [
                '北京今天的天气怎么样？',
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
    app_gui()          # 图形界面模式（默认）