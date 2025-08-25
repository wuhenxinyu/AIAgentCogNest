"""
ReAct：是一种“思考-行动-观察”循环范式（Reasoning + Acting），即
    • 模型会先思考（Thought），
    • 再决定调用哪个工具（Action），
    • 然后观察工具的输出（Observation），
    • 再继续推理，直到得出结论。
"""
import os
import json
import re
from typing import List, Union, Dict, Any, Optional

# 以下说明三个相关函数/类的区别：
# - `create_react_agent`: 专门用于创建基于 ReAct（Reasoning + Acting）范式的代理，
#   它会依据给定的工具和提示词模板创建一个能进行“思考-行动-观察”循环的代理。
# - `initialize_agent`: 一个更通用的函数，支持创建多种类型的代理，
#   通过 `agent` 参数指定代理类型（如 AgentType.ZERO_SHOT_REACT_DESCRIPTION），
#   能满足不同场景下的代理创建需求。
# - `AgentExecutor`: 代理执行器类，负责管理代理的执行流程，
#   接收代理和工具列表，控制代理与工具的交互过程并处理执行结果。
from langchain.agents import Tool, AgentExecutor, create_react_agent, AgentType, initialize_agent

from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain_community.llms import Tongyi
from langchain.memory import ConversationBufferMemory
import dashscope
# 从环境变量获取 dashscope 的 API Key
os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here'
api_key = os.environ.get('DASHSCOPE_API_KEY')
dashscope.api_key = api_key

# --- 环境设置 ---
# 确保设置了您的通义千问 API 密钥
# 您可以通过环境变量 DASHSCOPE_API_KEY 设置，或者直接在这里修改
# DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', '您的API密钥')
# 为方便演示，我们直接在此处硬编码 (请注意在生产环境中保护好您的密钥)
DASHSCOPE_API_KEY = 'sk-882e296067b744289acf27e6e20f3ec0' # 请替换为您的有效密钥

# --- 自定义网络诊断工具 ---
class PingTool:
    """执行 Ping 操作以检查与目标的网络连通性。"""
    def __init__(self):
        self.name = "网络连通性检查 (Ping)"
        self.description = "检查本机到指定主机名或 IP 地址的网络连通性。输入应该是目标主机名或 IP 地址。输出表明是否可达及延迟。"

    def run(self, target: str) -> str:
        """模拟执行 ping 命令。

        参数:
            target: 目标主机名或 IP 地址。
        返回:
            模拟的 ping 结果。
        """
        print(f"--- 模拟执行 Ping: {target} ---")
        # 简单模拟：特定目标可能失败
        if "unreachable" in target or target == "192.168.1.254":
            return f"Ping {target} 失败：请求超时。"
        elif target == "localhost" or target == "127.0.0.1":
             return f"Ping {target} 成功：延迟 <1ms。"
        elif "example.com" in target:
             # 模拟一个稍微慢点的响应
             import random
             delay = random.randint(20, 80)
             return f"Ping {target} 成功：延迟 {delay}ms。"
        else:
            # 其他情况默认成功
            import random
            delay = random.randint(5, 50)
            return f"Ping {target} 成功：延迟 {delay}ms。"

class DNSTool:
    """执行 DNS 查询以解析主机名。"""
    def __init__(self):
        self.name = "DNS解析查询"
        self.description = "解析给定的主机名，获取其对应的 IP 地址。输入应该是要解析的主机名。输出是 IP 地址或解析失败信息。"

    def run(self, hostname: str) -> str:
        """模拟 DNS 查询。

        参数:
            hostname: 要解析的主机名。
        返回:
            模拟的 DNS 解析结果。
        """
        print(f"--- 模拟 DNS 查询: {hostname} ---")
        # 简单模拟
        if hostname == "www.example.com":
            return f"DNS 解析 {hostname} 成功：IP 地址是 93.184.216.34"
        elif hostname == "internal.service.local":
            return f"DNS 解析 {hostname} 成功：IP 地址是 192.168.1.100"
        elif hostname == "unknown.domain.xyz":
            return f"DNS 解析 {hostname} 失败：找不到主机。"
        elif hostname == "127.0.0.1" or re.match(r"\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}", hostname):
             return f"输入 '{hostname}' 已经是 IP 地址，无需 DNS 解析。"
        else:
            # 模拟一个通用 IP
            return f"DNS 解析 {hostname} 成功：IP 地址是 10.0.0.5"

class InterfaceCheckTool:
    """检查本地网络接口的状态。"""
    def __init__(self):
        self.name = "本地网络接口检查"
        self.description = "检查本机网络接口的状态（如 IP 地址、是否启用）。可选输入是接口名称，若不提供则检查默认接口。输出接口状态信息。"

    def run(self, interface_name: Optional[str] = None) -> str:
        """模拟检查网络接口状态。

        参数:
            interface_name: (可选) 要检查的接口名称。
        返回:
            模拟的接口状态信息。
        """
        print(f"--- 模拟检查接口状态: {interface_name or '默认接口'} ---")
        if interface_name and "eth1" in interface_name.lower():
            return f"接口 '{interface_name}' 状态：关闭 (Administratively down)"
        else:
            # 模拟一个常见的有线或无线接口状态
            return f"接口 'Ethernet'/'Wi-Fi' 状态：启用, IP 地址: 192.168.1.50, 子网掩码: 255.255.255.0, 网关: 192.168.1.1"

class LogAnalysisTool:
    """分析系统或应用日志以查找网络相关的错误。"""
    def __init__(self):
        self.name = "网络日志分析"
        self.description = (
            "搜索系统或应用程序日志，查找与网络问题相关的条目。"
            "输入应该是描述问题的关键词（例如 'timeout', 'connection refused', 'dns error'）和可选的时间范围。"
            "输出是找到的相关日志条目摘要或未找到相关条目的消息。"
        )

    def run(self, keywords: str, time_range: Optional[str] = "过去1小时") -> str:
        """模拟分析日志。

        参数:
            keywords: 用于搜索日志的关键词。
            time_range: (可选) 要搜索的时间范围描述，默认为 '过去1小时'。
        返回:
            模拟的日志分析结果。
        """
        print(f"--- 模拟分析日志: 关键词='{keywords}', 时间范围='{time_range}' ---")
        # 简单模拟
        keywords_lower = keywords.lower()
        if "timeout" in keywords_lower or "超时" in keywords_lower:
            return (f"在 {time_range} 的日志中找到 3 条与 '{keywords}' 相关的条目：\n"
                    f"- [Error] 连接到 10.0.0.88:8080 超时\n"
                    f"- [Warning] 对 api.external.com 的请求超时\n"
                    f"- [Error] 内部服务通信超时")
        elif "connection refused" in keywords_lower or "连接被拒绝" in keywords_lower:
             return (f"在 {time_range} 的日志中找到 1 条与 '{keywords}' 相关的条目：\n"
                     f"- [Error] 连接到 192.168.1.200:5432 失败：Connection refused")
        elif "dns" in keywords_lower:
             return (f"在 {time_range} 的日志中找到 2 条与 '{keywords}' 相关的条目：\n"
                     f"- [Warning] DNS 服务器 8.8.8.8 响应慢\n"
                     f"- [Error] 无法解析主机名 'failed.internal.service'")
        else:
            return f"在 {time_range} 的日志中未找到与 '{keywords}' 相关的明显网络错误条目。"

# --- 创建 Agent 和工具链 ---

def create_network_diagnosis_chain():
    """创建网络故障诊断的 Agent 执行器。"""
    # 1. 初始化工具
    ping_tool = PingTool()
    dns_tool = DNSTool()
    interface_tool = InterfaceCheckTool()
    log_tool = LogAnalysisTool()

    # 2. 将工具类包装成 LangChain Tool 对象
    # 注意：这里的 description 对 Agent 如何使用工具至关重要
    tools = [
        Tool(
            name=ping_tool.name,
            func=ping_tool.run,
            description=ping_tool.description
        ),
        Tool(
            name=dns_tool.name,
            func=dns_tool.run,
            description=dns_tool.description
        ),
        Tool(
            name=interface_tool.name,
            func=interface_tool.run,
            description=interface_tool.description
        ),
        Tool(
            name=log_tool.name,
            func=log_tool.run,
            description=log_tool.description
        )
        # 如果有更多工具（如 Traceroute, Get Config等），在这里添加
    ]
    tool_names = ", ".join([t.name for t in tools])

    # 3. 初始化语言模型 (使用通义千问)
    llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=api_key)

    # 4. 创建 Zero-Shot ReAct Agent（简化版）
    """
    LangChain 框架内置的一种 Agent 类型，属于“Zero-Shot ReAct”范式。
    zero-shot-react-description：指的是大模型在没有额外训练或示例的情况下，直接根据提示词（Prompt）和工具描述来推理如何调用工具
    react-docstore: 用于与文档存储交互的 ReAct 代理类型，主要用于文档检索和问答场景，代理会根据问题进行推理，决定调用哪些工具与文档存储交互以获取答案。
    self-ask-with-search: 一种自询问搜索代理，代理会将复杂问题拆解为一系列子问题，并通过调用搜索工具逐步回答这些子问题，最终得到完整答案。
    conversational-react-description: 支持对话的 ReAct 代理，能够在对话场景中根据对话历史和工具描述，运用 ReAct 循环来决定调用哪些工具，适合需要上下文感知的对话任务。
    chat-zero-shot-react-description: 基于聊天模型的零样本 ReAct 代理，无需额外训练或示例，可以直接根据工具描述和输入内容在聊天场景中推理并调用工具。
    chat-conversational-react-description: 支持对话的聊天型 ReAct 代理，结合了聊天模型的优势和对话记忆，能在对话过程中利用历史对话内容和工具描述来决定工具调用，提供更连贯的对话体验。
    structured-chat-zero-shot-react-description: 结构化的聊天零样本 ReAct 代理，在聊天场景中，无需示例即可根据结构化的工具描述和输入内容进行推理并调用工具，适用于对输出格式有要求的场景。
    openai-functions: 基于 OpenAI 函数调用机制的代理类型，代理可以根据任务需求调用特定的函数，通过函数的输入输出与外部系统交互，常用于集成外部 API 或服务。
    openai-multi-functions: 支持多个函数调用的 OpenAI 代理类型，允许代理在一次交互中调用多个函数，适用于需要复杂操作或多步骤处理的任务。
    """
    """
      AgentType.ZERO_SHOT_REACT_DESCRIPTION 的特点：
        1、自动读取你传入的工具（Tool）列表，每个工具的name 和 description 会被自动拼接到系统提示词中。
        2、Agent 会根据用户输入和工具描述，自动决定调用哪个工具、如何调用。
        3、你不需要自己写复杂的 Prompt，LangChain 内部会.自动生成一份标准的英文提示词。
    """
    # 使用 initialize_agent 函数创建一个 Agent 执行器，该执行器能够根据输入自动调用合适的工具进行推理和处理
    # 参数说明：
    # tools: 传入之前创建好的工具列表，这些工具定义了 Agent 可以执行的具体操作，如 Ping 检查、DNS 解析等
    # llm: 传入之前初始化好的语言模型，这里使用的是通义千问模型，用于生成推理和决策
    # agent: 指定 Agent 的类型为 ZERO_SHOT_REACT_DESCRIPTION，这意味着 Agent 可以在没有额外训练的情况下，
    #        根据工具的描述和输入内容决定如何调用工具
    # verbose: 设置为 True 表示会打印出 Agent 的思考和行动过程，方便调试和观察
    # handle_parsing_errors: 设置为 False 表示不自动处理解析错误，遇到错误会直接抛出异常
    # memory: 使用 ConversationBufferMemory 来维护对话历史，将对话历史存储在 "chat_history" 键中，
    #         输入键为 "input"，输出键为 "output"，使得 Agent 可以参考之前的对话内容
    # max_iterations: 设置 Agent 最大的迭代次数为 10，防止无限循环
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=False,
        memory=ConversationBufferMemory(memory_key="chat_history", input_key="input", output_key="output"),
        max_iterations=10
    )

    return agent

# --- 示例：使用 Agent 处理网络诊断任务 ---

def diagnose_network_issue(issue_description: str):
    """
    使用网络诊断 Agent 处理用户报告的网络问题。

    参数:
        issue_description: 用户描述的网络问题。
    返回:
        Agent 的最终诊断结果。
    """
    try:
        print(f"\\n--- 开始诊断任务 ---")
        print(f"用户问题: {issue_description}")
        agent = create_network_diagnosis_chain()
        # 使用 invoke 而不是 run，以便更好地处理输入输出和记忆
        response = agent.invoke({"input": issue_description})
        # response 是一个字典，我们通常关心 "output" 键
        return response.get("output", "未能获取诊断结果。")
    except Exception as e:
        # 打印更详细的错误信息，有助于调试
        import traceback
        print(f"处理诊断任务时发生错误: {str(e)}")
        traceback.print_exc()
        return f"处理诊断任务时出错: {str(e)}"

# --- 主程序入口 ---
if __name__ == "__main__":
    # 示例 1: 无法访问特定网站
    task1 = "我无法访问 www.example.com，浏览器显示连接超时。"
    print("诊断任务 1:")
    result1 = diagnose_network_issue(task1)
    print("\\n--- 诊断任务 1 结束 ---")
    print(f"最终诊断结果: {result1}")

    print("\\n" + "="*50 + "\\n") # 分隔符

    # 示例 2: 内部服务访问失败
    task2 = "连接到内部数据库服务器 (internal.service.local) 失败，提示 'connection refused'。"
    print("诊断任务 2:")
    result2 = diagnose_network_issue(task2)
    print("\\n--- 诊断任务 2 结束 ---")
    print(f"最终诊断结果: {result2}")

    # # 示例 3: DNS 解析问题 (需要 DNSTool 模拟失败)
    # task3 = "我打不开网站 unknown.domain.xyz，好像是 DNS 问题。"
    # print("诊断任务 3:")
    # result3 = diagnose_network_issue(task3)
    # print("\\n--- 诊断任务 3 结束 ---")
    # print(f"最终诊断结果: {result3}")

    # # 示例 4: 本地网络接口问题 (需要 InterfaceCheckTool 模拟失败)
    # task4 = "我的电脑连不上网了，检查一下接口 eth1 的状态。"
    # print("诊断任务 4:")
    # result4 = diagnose_network_issue(task4)
    # print("\\n--- 诊断任务 4 结束 ---")
    # print(f"最终诊断结果: {result4}") 