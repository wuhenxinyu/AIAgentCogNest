from openai import OpenAI
import os
import base64
os.environ['DASHSCOPE_API_KEY'] = 'sk-siRF78nIxVVBKekhvZF6POAzSrFXymXwzCFj4YT6SzFIlvWA'
api_key = os.getenv("DASHSCOPE_API_KEY")

local_file_path = '13-RAG技术与应用/图片理解/6-万圣节.jpeg'

# 1. 读取本地图片并转为 Base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

messages = [{
    'role': 'system',
    'content': '你是一个乐于助人的助手。'
}, {
    'role': 'user',
    'content': [
        {
            'type': 'image_url',
            # 直接使用 Base64 数据
             "image_url": f"data:image/jpeg;base64,{image_to_base64(local_file_path)}"
        },
        {
            'type': 'text',
            'text': '这是一张什么海报？'
        }
    ]
}]

client = OpenAI(
        api_key=api_key,
        base_url="https://api.fe8.cn/v1"
    )

# 调用通义千问大模型的多模态对话接口
# model参数指定使用qwen-vl-plus模型
# messages参数包含了系统提示词和用户输入的图片及文本
response = client.chat.completions.create(
    model="qwen-vl-plus",
    messages=messages,
    max_tokens=300
)
print(response)