import pprint
import urllib.parse
import json5
from .agents import Assistant
from .tools.base import BaseTool, register_tool
import os
from .gui import WebUI
os.environ['DASHSCOPE_API_KEY'] = 'sk-58f051ae745e4bb19fdca31735105b11'

# 步骤 1（可选）：添加一个名为 `my_image_gen` 的自定义工具。
@register_tool('my_image_gen')
class MyImageGen(BaseTool):
    # `description` 用于告诉智能体该工具的功能。
    description = 'AI 绘画（图像生成）服务，输入文本描述，返回基于文本信息绘制的图像 URL。'
    # `parameters` 告诉智能体该工具有哪些输入参数。
    parameters = [{
        'name': 'prompt',
        'type': 'string',
        'description': '期望的图像内容的详细描述',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        # `params` 是由 LLM 智能体生成的参数。
        prompt = json5.loads(params)['prompt']
        prompt = urllib.parse.quote(prompt)
        return json5.dumps(
            {'image_url': f'https://image.pollinations.ai/prompt/{prompt}'},
            ensure_ascii=False)


def init_agent_service():
    """初始化助手服务"""
    # 步骤 2：配置您所使用的 LLM。
    # llm_cfg = {
    #     # 使用 DashScope 提供的模型服务：
    #     'model': 'qwen-max',
    #     'model_server': 'dashscope',
    #     'api_key': os.getenv('DASHSCOPE_API_KEY'),  # 从环境变量获取API Key
    #     'generate_cfg': {
    #         'top_p': 0.8
    #     }
    # }

    llm_cfg = {
        # 使用 DashScope 提供的模型服务：
        'model': 'qwen-max',
        'model_server': 'dashscope',
        'api_key': os.getenv('DASHSCOPE_API_KEY'),  # 从环境变量获取API Key
        'generate_cfg': {
            'top_p': 0.8
        }
    }

    # 步骤 3：创建一个智能体。这里我们以 `Assistant` 智能体为例，它能够使用工具并读取文件。
    system_instruction = '''你是一个乐于助人的AI助手。
在收到用户的请求后，你应该：
- 首先绘制一幅图像，得到图像的url，
- 然后运行代码`request.get`以下载该图像的url，
- 最后从给定的文档中选择一个图像操作进行图像处理。
用 `plt.show()` 展示图像。
你总是用中文回复用户。'''
    tools = ['my_image_gen', 'code_interpreter']  # `code_interpreter` 是框架自带的工具，用于执行代码。
    # 获取文件夹下所有文件
    file_dir = os.path.join(os.path.dirname(__file__), 'docs')
    files = []
    if os.path.exists(file_dir):
        # 遍历目录下的所有文件
        for file in os.listdir(file_dir):
            file_path = os.path.join(file_dir, file)
            if os.path.isfile(file_path):  # 确保是文件而不是目录
                files.append(file_path)
    print('files=', files)

    bot = Assistant(llm=llm_cfg,
                    system_message=system_instruction,
                    function_list=tools,
                    files=files)
    return bot


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
                
                # 输入验证
                if not query:
                    print('user question cannot be empty！')
                    continue
                    
                # 构建消息
                messages.append({'role': 'user', 'content': query})

                print("正在处理您的请求...")
                # 运行助手并处理响应
                response = []
                current_index = 0
                first_chunk = True
                for response_chunk in bot.run(messages=messages):
                    if first_chunk:
                        # 尝试获取并打印召回的文档内容
                        if hasattr(bot, 'retriever') and bot.retriever:
                            print("\n===== 召回的文档内容 =====")
                            retrieved_docs = bot.retriever.retrieve(query)
                            if retrieved_docs:
                                for i, doc in enumerate(retrieved_docs):
                                    print(f"\n文档片段 {i+1}:")
                                    print(f"内容: {doc.page_content}")
                                    print(f"元数据: {doc.metadata}")
                            else:
                                print("没有召回任何文档内容")
                            print("===========================\n")
                        first_chunk = False

                    # The response is a list of messages. We are interested in the assistant's message.
                    if response_chunk and response_chunk[0]['role'] == 'assistant':
                        assistant_message = response_chunk[0]
                        new_content = assistant_message.get('content', '')
                        print(new_content[current_index:], end='', flush=True)
                        current_index = len(new_content)
                    
                    response = response_chunk
                
                print() # New line after streaming.

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
                '画一只在写代码的猫',
                '介绍下雇主责任险',
                '帮我画一个宇宙飞船，然后把它变成黑白的'
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
    # app_tui() 