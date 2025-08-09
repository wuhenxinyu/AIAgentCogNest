import json
import os
import dashscope
from dashscope.api_entities.dashscope_response import Role
from dashscope import MultiModalConversation
import os
os.environ['DASHSCOPE_API_KEY'] = 'sk-58f051ae745e4bb19fdca31735105b11'
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

local_file_path = 'file://13-RAG技术与应用/图片理解/6-万圣节.jpeg'
messages = [{
    'role': 'system',
    'content': [{
        'text': '你是一个乐于助人的助手。'
    }]
}, {
    'role': 'user',
    'content': [
        {
            'image': local_file_path
        },
        {
            'text': '这是一张什么海报？'
        },
    ]
}]
# 调用通义千问大模型的多模态对话接口
# model参数指定使用qwen-vl-plus模型
# messages参数包含了系统提示词和用户输入的图片及文本
response = MultiModalConversation.call(model='qwen-vl-plus', messages=messages)
print(response)