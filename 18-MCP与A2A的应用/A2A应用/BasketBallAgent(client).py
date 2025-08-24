"""
    篮球智能体客户端：
    1 发现：客户端从 /.well-known/agent.json 获取 Agent Card，了解智能体能力。
    2 启动：客户端发送任务请求：
        • 使用 tasks/send 处理即时任务，返回最终 Task 对象。
        • 使用 tasks/sendSubscribe 处理长期任务，服务器通过 SSE 事件发送更新。
    3 处理：服务器处理任务，可能涉及流式更新或直接返回结果。
    4 交互（可选）：若任务状态为 input-required，客户端可发送更多消息，使用相同 Task ID 提供输入。
    5 完成：任务达到终端状态（如 completed、failed 或 canceled）。
"""
import requests
import uuid

class BasketBallAgent:
    def __init__(self):
        self.weather_agent_url = "http://localhost:8000"
        self.api_key = "SECRET_KEY"  # 实际应通过安全方式存储

    def _create_task(self, target_date: str) -> dict:
        """创建A2A标准任务对象"""
        return {
            "task_id": str(uuid.uuid4()),
            "params": {
                "date": target_date,
                "location": "北京"
            }
        }

    def check_weather(self, target_date: str) -> dict:
        """通过A2A协议查询天气"""
        # 获取天气智能体能力描述
        agent_card = requests.get(
            f"{self.weather_agent_url}/.well-known/agent.json"
        ).json()
        # 构造任务请求
        task = self._create_task(target_date)
        # 发送任务请求
        response = requests.post(
            f"{self.weather_agent_url}{agent_card['endpoints']['task_submit']}",
            json=task,
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        if response.status_code == 200:
            return response.json()["artifact"]
        else:
            raise Exception(f"天气查询失败: {response.text}")

    def schedule_meeting(self, date: str):
        """综合决策逻辑"""
        try:
            result = self.check_weather(date)
            # print('result=', result)  
            if "雨" not in result["weather"]["condition"] and "雪" not in result["weather"]["condition"]:
                return {"status": "confirmed", "weather": result["weather"]}
            else:
                return {"status": "cancelled", "reason": "恶劣天气"}
        except Exception as e:
            return {"status": "error", "detail": str(e)}

# 使用示例
if __name__ == "__main__":
    meeting_agent = BasketBallAgent()
    # result = meeting_agent.schedule_meeting("2025-05-08")
    result = meeting_agent.schedule_meeting("2025-05-10")
    print("篮球安排结果:", result)