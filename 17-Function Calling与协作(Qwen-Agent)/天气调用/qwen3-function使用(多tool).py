import requests
from http import HTTPStatus
import dashscope

# 设置 DashScope API Key
dashscope.api_key = "你的API KEY"

# 高德天气 API 的 天气工具定义（JSON 格式）
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. 北京",
                },
                "adcode": {
                    "type": "string",
                    "description": "The city code, e.g. 110000 (北京)",
                }
            },
            "required": ["location"],
        },
    },
}

def get_weather_from_gaode(location: str, adcode: str = None):
    """调用高德地图API查询天气"""
    gaode_api_key = "你的API KEY"  # 替换成你的高德API Key
    base_url = "https://restapi.amap.com/v3/weather/weatherInfo"
    
    params = {
        "key": gaode_api_key,
        "city": adcode if adcode else location,
        "extensions": "base",  # 可改为 "all" 获取预报
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to fetch weather: {response.status_code}"}

def run_weather_query():
    """使用 Qwen3 + 查询天气，并让大模型输出最终结果"""
    messages = [
        {"role": "system", "content": "你是一个智能助手，可以查询天气信息。"},
        {"role": "user", "content": "北京现在天气怎么样？"}
    ]
    
    print("第一次调用大模型...")
    response = dashscope.Generation.call(
        model="qwen-turbo",  # 可使用 Qwen3 最新版本
        messages=messages,
        tools=[weather_tool],  # 传入工具定义
        tool_choice="auto",  # 让模型决定是否调用工具
    )
    
    if response.status_code == HTTPStatus.OK:
        tool_map = {
            "get_current_weather": get_weather_from_gaode,
            # 如有更多工具，在此添加
        }
        
        # 从响应中获取消息
        assistant_message = response.output.choices[0].message
        
        # 检查是否需要调用工具
        if hasattr(assistant_message, "tool_calls") and assistant_message.tool_calls:
            print("检测到工具调用...")
            
            # 转换 assistant 消息为标准字典格式
            assistant_dict = {
                "role": "assistant",
                "content": assistant_message.content if hasattr(assistant_message, "content") else None
            }
            
            # 添加 tool_calls 到 assistant 消息
            if hasattr(assistant_message, "tool_calls"):
                assistant_dict["tool_calls"] = assistant_message.tool_calls
                
                # 生成工具调用回复消息
                tool_response_messages = []
                import json
                for tool_call in assistant_message.tool_calls:
                    print(f"处理工具调用: {tool_call['function']['name']}, ID: {tool_call['id']}")
                    
                    func_name = tool_call["function"]["name"]
                    func_args = json.loads(tool_call["function"]["arguments"])
                    
                    if func_name in tool_map:
                        # 调用工具函数
                        from inspect import signature
                        sig = signature(tool_map[func_name])
                        valid_args = {k: v for k, v in func_args.items() if k in sig.parameters}
                        result = tool_map[func_name](**valid_args)
                        
                        # 创建工具回复消息
                        tool_response = {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": func_name,
                            "content": json.dumps(result, ensure_ascii=False)
                        }
                        tool_response_messages.append(tool_response)
                
                # 组装完整消息列表
                updated_messages = messages + [assistant_dict] + tool_response_messages
                
                print(f"完整消息列表: {updated_messages}")
                
                # 第二次调用大模型
                print("第二次调用大模型...")
                # 第二次调用大模型，将第一次调用后生成的完整消息列表 `updated_messages` 传入，
                # 该列表包含了初始消息、助手消息以及工具调用的响应消息。
                # 使用 `qwen-turbo` 模型，同时传入天气工具定义 `weather_tool`，
                # 设置 `tool_choice` 为 "auto"，让模型自动决定是否需要再次调用工具。
                # 此次调用的目的是让大模型根据工具调用的结果生成最终的用户可见回复。
                response2 = dashscope.Generation.call(
                    model="qwen-turbo",  # 指定使用的大模型为 qwen-turbo
                    messages=updated_messages,  # 传入包含工具调用结果的完整消息列表
                    tools=[weather_tool],  # 传入天气工具定义
                    tool_choice="auto",  # 让模型自动决定是否调用工具
                )
                if response2.status_code == HTTPStatus.OK:
                    final_response = response2.output.choices[0].message.content
                    print("最终回复:", final_response)
                else:
                    print(f"请求失败: {response2.code} - {response2.message}")
            else:
                print("assistant 消息中没有 tool_calls 字段")
                print(assistant_message)
        else:
            # 如果没有调用工具，直接输出模型回复
            print("无工具调用，直接输出回复:", assistant_message.content)
    else:
        print(f"请求失败: {response.code} - {response.message}")

if __name__ == "__main__":
    run_weather_query()