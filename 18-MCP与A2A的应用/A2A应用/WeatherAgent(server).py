"""
天气智能体服务端

A2A 与MCP 是互补的，可以将A2A看成是一个电话簿，MCP看成是工具说明书。MCP 确保Agent能访问数据和工具（如通过 Google Drive、Slack、GitHub 等），A2A 则让Agent能协作处理这些数据，完成任务。
    1、MCP（模型上下文协议）用于工具和资源
        • 通过结构化的输入/输出将代理连接到工具、API 和资源。
        • Google ADK 支持 MCP 工具，允许与Agent一起使用广泛的 MCP 服务器。
    2、A2A（Agent2Agent 协议）用于Agent之间的协作
        • 在不同Agent之间实现动态的、多模态的通信，而无需共享内存、资源和工具。
        • 这是一个由社区驱动的开放标准。
        • 在 Google ADK、LangGraph 和 Crew.AI 中有提供的参考示例
    3、MCP 与 A2A 的结合
        • MCP 提供了一种标准化的方式，使不同的智能体能够相互通信和协作。
        • A2A 则提供了一种动态的、多模态的通信机制，使智能体能够处理复杂的任务。
        • 这两者的结合使得智能体能够更加强大，能够处理更复杂的任务。    
    4、MCP 与 A2A 的应用场景
        • 跨平台协作：不同平台的智能体可以通过 MCP 进行通信，实现跨平台协作。
        • 多智能体系统：多个智能体可以通过 A2A 进行协作，实现复杂的任务处理。
        • 混合智能体系统：结合 MCP 和 A2A，智能体可以同时利用 MCP 调用工具和 A2A 进行协作。
        • 智能体与用户交互：智能体可以通过 A2A 与用户进行交互，接收用户输入并执行相应的任务。
        • 智能体与其他智能体交互：智能体可以通过 A2A 与其他智能体进行交互，共享信息和资源。
        • 智能体与外部系统交互：智能体可以通过 MCP 调用外部系统的 API，实现与外部系统的集成。

协议的交集：
    Google建议应用将A2A智能体通过AgentCard描述注册为MCP资源。这样，框架既能通过MCP调用工具，又能通过A2A与用户、远程智能体通信，实现无缝协作。
    例如，一个智能体可能通过 MCP 从数据库检索数据，然后通过 A2A 与另一个智能体协作分析数据。    
"""
from fastapi import FastAPI, HTTPException
from datetime import date
from pydantic import BaseModel
import uvicorn
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WeatherAgent")

app = FastAPI()

# Agent Card声明（通过/.well-known/agent.json暴露智能体能力描述）
# 标准化任务交互：使用统一的任务提交和结果获取格式
# 身份验证：通过API Key进行授权
WEATHER_AGENT_CARD = {
    "name": "WeatherAgent",
    "version": "1.0",
    "description": "提供指定日期的天气数据查询",
    "endpoints": {
        "task_submit": "/api/tasks/weather",
        "sse_subscribe": "/api/tasks/updates"
    },
    "input_schema": {
        "type": "object",
        "properties": {
            "date": {"type": "string", "format": "date"},
            "location": {"type": "string", "enum": ["北京"]}
        },
        "required": ["date"]
    },
    "authentication": {"methods": ["API_Key"]}
}

# 任务请求模型
class WeatherTaskRequest(BaseModel):
    task_id: str
    params: dict

# 模拟天气数据存储
weather_db = {
    "2025-05-08": {"temperature": "25℃", "condition": "雷阵雨"},
    "2025-05-09": {"temperature": "18℃", "condition": "小雨转晴"},
    "2025-05-10": {"temperature": "22℃", "condition": "多云转晴"}
}

@app.get("/.well-known/agent.json")
async def get_agent_card():
    logger.info("Agent card requested")
    return WEATHER_AGENT_CARD

@app.post("/api/tasks/weather")
async def handle_weather_task(request: WeatherTaskRequest):
    """处理天气查询任务"""
    logger.info(f"Received weather task request: task_id={request.task_id}, params={request.params}")
    target_date = request.params.get("date")
    # 参数验证
    if not target_date or target_date not in weather_db:
        logger.warning(f"Invalid date parameter: {target_date}")
        raise HTTPException(status_code=400, detail="无效日期参数")
    
    result = {
        "task_id": request.task_id,
        "status": "completed",
        "artifact": {
            "date": target_date,
            "weather": weather_db[target_date]
        }
    }
    
    logger.info(f"Weather task completed: {result}")
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)