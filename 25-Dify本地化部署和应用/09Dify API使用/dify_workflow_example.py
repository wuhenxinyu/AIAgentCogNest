"""
Dify 工作流应用调用示例
专门针对工作流类型的应用
"""

from dify_agent_client import DifyAgentClient

def simple_workflow_example():
    """简单的工作流调用示例"""
    
    # 您的Dify配置
    BASE_URL = "https://api.dify.ai/v1"
    API_KEY = "app-95EbA6qI7aE7BqkvK0Tue1p1"
    
    # 创建客户端
    client = DifyAgentClient(BASE_URL, API_KEY)
    
    user_input = "离离原上草"
    print(f"发送消息: {user_input}")
    
    result = client.run_workflow(
        inputs={"input": user_input},
        user_id="demo_user"
    )
    
    if result.get("error"):
        print(f"调用失败: {result.get('message')}")
    else:
        print(f"完整回复: {result}")
        print(f"工作流回复: {result.get('answer')}")
        print(f"工作流运行ID: {result.get('workflow_run_id')}")


if __name__ == "__main__":
    simple_workflow_example()
