
import json
import os
import dashscope
from dashscope.api_entities.dashscope_response import Role
from dashscope.common.constants import DASHSCOPE_API_KEY_ENV

# 阿里云百炼智能开放平台申请API Key
os.environ[DASHSCOPE_API_KEY_ENV] = "your_api_key_here"
# 从环境变量中，获取 DASHSCOPE_API_KEY
api_key = os.environ.get(DASHSCOPE_API_KEY_ENV)
dashscope.api_key = api_key

# 封装模型响应函数
def get_response(messages):
    # 调用 dashscope 的多模态对话接口，用于处理包含图片和文本的多模态对话任务，支持图片理解、文本生成等能力
    response = dashscope.MultiModalConversation.call(
        # 使用通义千问VL-Plus大模型，支持多模态对话，可以处理图像理解和文本生成任务
        model='qwen-vl-plus',
        messages=messages
    )
    return response

content = [
    {'image': 'https://aiwucai.oss-cn-huhehaote.aliyuncs.com/pdf_table.jpg'}, # Either a local path or an url
    {'text': '这是一个表格图片，帮我提取里面的内容，输出JSON格式'}
]

messages=[{"role": "user", "content": content}]
# 得到响应
response = get_response(messages)
print(response)

print('message:', response.output.choices[0].message.content[0]['text'])

