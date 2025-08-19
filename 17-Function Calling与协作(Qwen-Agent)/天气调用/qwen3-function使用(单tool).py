import requests
from http import HTTPStatus
import dashscope

# 设置 DashScope API Key
dashscope.api_key = "your_dashscope_api_key"

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
    """使用 Qwen3 + 查询天气"""
    messages = [
        {"role": "system", "content": "你是一个智能助手，可以查询天气信息。"},
        {"role": "user", "content": "北京现在天气怎么样？"}
    ]
    
    response = dashscope.Generation.call(
        model="qwen-turbo",  # 可使用 Qwen3 最新版本
        messages=messages,
        tools=[weather_tool],  # 传入工具定义
        tool_choice="auto",  # 让模型决定是否调用工具
    )
    
    if response.status_code == HTTPStatus.OK:
        # 检查是否需要调用工具
        if "tool_calls" in response.output.choices[0].message:
            tool_call = response.output.choices[0].message.tool_calls[0]
            if tool_call["function"]["name"] == "get_current_weather":
                # 解析参数并调用高德API
                import json
                args = json.loads(tool_call["function"]["arguments"])
                location = args.get("location", "嘉定区")
                adcode = args.get("adcode", None)
                
                weather_data = get_weather_from_gaode(location, adcode)
                print(f"查询结果：{weather_data}")
        else:
            print(response.output.choices[0].message.content)
    else:
        print(f"请求失败: {response.code} - {response.message}")

if __name__ == "__main__":
    run_weather_query()