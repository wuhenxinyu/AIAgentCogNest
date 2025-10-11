"""
多智能体问答系统,利用 Qwen Agent 实现多智能体协作与 @mention（@提及）功能:
    Step1，定义多智能体,包含代码解释器（ReActChat）、文档问答（BasicDocQA）、通用助手（Assistant）三类智能体。
    Step2，初始化服务,通过 init_agent_service 返回智能体列表。
    Step3，多智能体协作
        • 用户输入被解析后，自动路由到对应的智能体进行处理。
        • 支持多轮对话和多智能体并存，每个智能体有独立的状态管理。
        • 支持 @mention 功能，用户可以在输入中 @mention 不同的智能体，实现 targeted 对话。
"""

from qwen_agent.agents import Assistant, ReActChat
from qwen_agent.agents.doc_qa import BasicDocQA
from qwen_agent.gui import WebUI
import dashscope
import os

dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', 'sk-58f051ae745e4bb19fdca31735105b11')


"""
多智能体初始化与注册
统一初始化多种类型智能体，便于后续协作和路由。
"""
def init_agent_service():
    llm_cfg = {'model': 'qwen-max'}
    # 代码解释器（ReActChat）
    """
    ReActChat 的适用场景：
    • 适用于需要 工具调用+推理 场景，如代码解释、数据分析等。
    • 支持 function_list 配置（如 code_interpreter），可自动调用外部工具。
    • 支持 system_message 指定智能体角色和风格。
    """
    react_chat_agent = ReActChat(
        llm=llm_cfg,
        name='代码解释器',
        description='代码解释器，可用于执行Python代码。',
        system_message='you are a programming expert, skilled in writing code '
        'to solve mathematical problems and data analysis problems.',
        function_list=['code_interpreter'],
    )
    # 文档问答（BasicDocQA）
    """
    BasicDocQA 的适用场景：
    • 适用于需要根据文档内容回答问题的场景，支持根据用户问题和上传文档自动检索答案。如文档问答、知识检索等。
    • 支持 document_list 配置，可指定多个文档源。
    • 支持 system_message 指定智能体角色和风格。
    """
    doc_qa_agent = BasicDocQA(
        llm=llm_cfg,
        name='文档问答',
        description='根据用户输入的问题和文档，从文档中找到答案',
    )
    # 通用助手（Assistant）
    assistant_agent = Assistant(llm=llm_cfg, name='小助理', description="I'm a helpful assistant")

    return [react_chat_agent, doc_qa_agent, assistant_agent]


def app_gui():
    agent_list = init_agent_service()
    chatbotConfig = {
        'prompt.suggestions': [
            '@代码解释器 2 ^ 10 = ?',
            '@文档问答 这篇论文解决了什么问题？',
            '@小助理 你好！',
        ],
        'verbose': True
    }
    WebUI(
        agent_list,
        chatbot_config=chatbotConfig,
    ).run(messages=[{
        'role': 'assistant',
        'content': [{
            'text': '试试看 @代码解释器 来问我~'
        }]
    }], enable_mention=True)


if __name__ == '__main__':
    app_gui()
