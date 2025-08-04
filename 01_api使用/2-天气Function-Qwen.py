
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

# 编写你的天气函数
# 为了演示流程，这里指定了天气的温度，实际上可以调用 高德接口获取实时天气。
# 这里可以先用每个城市的固定天气进行返回，查看大模型的调用情况
def get_current_weather(location, unit="摄氏度"):
    # 获取指定地点的天气
    temperature = -1
    if '大连' in location or 'Dalian' in location:
        temperature = 10
    if '上海' in location or 'Shanghai' in location:
        temperature = 36
    if '深圳' in location or 'Shenzhen' in location:
        temperature = 37
    weather_info = {
        "location": location,
        "temperature": temperature,
        "unit": unit,
        "forecast": ["晴天", "微风"],
    }
    # 将天气信息字典转换为JSON字符串返回,方便后续处理和传输
    return json.dumps(weather_info)

# 封装模型响应函数
def get_response(messages):
    try:
        response = dashscope.Generation.call(
            model='qwen-turbo',# model: 使用qwen-turbo模型
            messages=messages, # messages: 传入对话历史消息
            functions=functions, # functions: 传入可调用的函数列表
            result_format='message', # result_format: 指定返回结果格式为message
        )
        return response
    except Exception as e:
        print(f"API调用出错: {str(e)}")
        return None

# 使用function call进行QA
def run_conversation():
    query = "上海的天气怎样"
    messages=[{"role": "user", "content": query}]
    
    # 得到第一次响应
    response = get_response(messages)
    if not response or not response.output:
        print("获取响应失败")
        return None
        
    print('response=', response)
    
    message = response.output.choices[0].message
    messages.append(message)
    print('message=', message)
    
    # Step 2, 判断用户是否要call function
    # 检查消息对象是否包含function_call属性且该属性不为空
    # 如果模型决定调用函数，这个条件会为True
    if hasattr(message, 'function_call') and message.function_call:
        function_call = message.function_call
        tool_name = function_call['name']
        # Step 3, 执行function call
        arguments = json.loads(function_call['arguments'])
        print('arguments=', arguments)
        tool_response = get_current_weather(
            location=arguments.get('location'),
            unit=arguments.get('unit'),
        )
        tool_info = {"role": "function", "name": tool_name, "content": tool_response}
        print('tool_info=', tool_info)
        messages.append(tool_info)
        print('messages=', messages)
        
        #Step 4, 得到第二次响应
        response = get_response(messages)
        if not response or not response.output:
            print("获取第二次响应失败")
            return None
            
        print('response=', response)
        message = response.output.choices[0].message
        return message
    return message

# 这个地方的description
functions = [
    {
      'name': 'get_current_weather',
      'description': '获取指定位置的当前天气信息',
      'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': '城市名称，例如：北京、上海、深圳'
                },
                'unit': {
                    'type': 'string', 
                    'enum': ['摄氏度', '华氏度'],
                    'description': '温度单位，默认为摄氏度'
                }
            },
        'required': ['location']
      }
    }
]

if __name__ == "__main__":
    result = run_conversation()
    if result:
        print("最终结果:", result)
    else:
        print("对话执行失败")

