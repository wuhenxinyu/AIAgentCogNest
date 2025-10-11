"""多智能体协作路由示例，演示如何通过路由器和助手实现多智能体协作"""

import os
from typing import Optional

from qwen_agent.agents import Assistant, ReActChat, Router
from qwen_agent.gui import WebUI

import dashscope
import os

dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', 'sk-58f051ae745e4bb19fdca31735105b11')

ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')


def init_agent_service():
    # 配置大模型
    llm_cfg = {'model': 'qwen-max'}
    llm_cfg_vl = {'model': 'qwen-vl-max'}
    tools = ['image_gen', 'code_interpreter']

    # 定义多模态助手
    bot_vl = Assistant(
        llm=llm_cfg_vl, 
        name='多模态助手', 
        description='可以理解图像内容。'
    )

    # 定义工具助手
    bot_tool = ReActChat(
        llm=llm_cfg,
        name='工具助手',
        description='可以使用画图工具和运行代码来解决问题',
        function_list=tools,
    )

    # 定义路由器
    bot = Router(
        llm=llm_cfg,
        agents=[bot_vl, bot_tool],
    )
    return bot


def test(
        query: str = '你好',
        image: str = 'https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg',
        file: Optional[str] = os.path.join(ROOT_RESOURCE, 'poem.pdf'),
):
    # 初始化多智能体路由
    bot = init_agent_service()

    # 构造对话消息
    messages = []

    if not image and not file:
        messages.append({'role': 'user', 'content': query})
    else:
        messages.append({'role': 'user', 'content': [{'text': query}]})
        if image:
            messages[-1]['content'].append({'image': image})
        if file:
            messages[-1]['content'].append({'file': file})

    for response in bot.run(messages):
        print('bot response:', response)


def app_tui():
    # 初始化多智能体路由
    bot = init_agent_service()

    # 命令行对话
    messages = []
    while True:
        query = input('请输入你的问题: ')
        # 图片示例: https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg
        image = input('图片URL（如无可留空）: ')
        # 文件示例: resource/poem.pdf
        file = input('文件URL（如无可留空）: ').strip()
        if not query:
            print('用户问题不能为空！')
            continue
        if not image and not file:
            messages.append({'role': 'user', 'content': query})
        else:
            messages.append({'role': 'user', 'content': [{'text': query}]})
            if image:
                messages[-1]['content'].append({'image': image})
            if file:
                messages[-1]['content'].append({'file': file})

        response = []
        for response in bot.run(messages):
            print('bot response:', response)
        messages.extend(response)


def app_gui():
    bot = init_agent_service()
    chatbot_config = {
        # verbose 设置为 True 表示开启详细日志输出，便于调试和查看交互过程中的详细信息
        'verbose': True,
    }
    WebUI(bot, chatbot_config=chatbot_config).run()


if __name__ == '__main__':
    # test()
    # app_tui()
    app_gui() 