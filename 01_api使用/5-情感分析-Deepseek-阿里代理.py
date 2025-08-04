import dashscope
from dashscope.api_entities.dashscope_response import Role
import os
from dashscope.common.constants import DASHSCOPE_API_KEY_ENV

# 阿里云百炼智能开放平台申请API Key
os.environ[DASHSCOPE_API_KEY_ENV] = "your_api_key_here"
# 从环境变量中，获取 DASHSCOPE_API_KEY
api_key = os.environ.get(DASHSCOPE_API_KEY_ENV)
dashscope.api_key = api_key

# 封装模型响应函数
def get_response(messages):
    response = dashscope.Generation.call(
        model='deepseek-r1',  # 使用 deepseek-r1 模型
        messages=messages,
        result_format='message'  # 将输出设置为message形式
    )
    return response

# 初始化对话历史
messages = [
    {"role": "system", "content": "You are a helpful assistant"}
]

# 第一次对话
user_input = "你好，你是什么大模型？"
messages.append({"role": "user", "content": user_input})
response = get_response(messages)
assistant_response = response.output.choices[0].message.content
print(f"用户: {user_input}")
print(f"助手: {assistant_response}")

# 将助手回复添加到对话历史
messages.append({"role": "assistant", "content": assistant_response})

# 继续对话的循环
print("\n开始多轮对话（输入'退出'结束对话）:")
while True:
    user_input = input("\n用户: ")
    if user_input.lower() in ["退出", "exit", "quit", "q"]:
        print("对话已结束")
        break
    
    # 添加用户输入到对话历史
    messages.append({"role": "user", "content": user_input})
    
    # 获取模型回复
    response = get_response(messages)
    assistant_response = response.output.choices[0].message.content
    print(f"助手: {assistant_response}")
    
    # 将助手回复添加到对话历史
    messages.append({"role": "assistant", "content": assistant_response})

