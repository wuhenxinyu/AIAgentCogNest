import json
import os
from re import M, S
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
    response = dashscope.Generation.call(
        model='deepseek-v3',
        messages=messages,
        result_format='message',  # 将输出设置为message形式
        stream=True,  # 开启流式输出
        temperature=0.7, # 温度参数，控制随机性
        top_p=0.8, # 控制输出的多样性
        max_tokens=1500, # 最大输出长度
    )
    return response
    
review = '这款音效特别好 给你意想不到的音质。'
messages=[
    {"role": "system", "content": "你是一名舆情分析师，帮我判断产品口碑的正负向，回复请用一个词语：正向 或者 负向"},
    {"role": "user", "content": review}
  ]

response = get_response(messages)
# 如果是流式输出
for chunk in response:
    print(chunk.output.choices[0].message.content, end='')