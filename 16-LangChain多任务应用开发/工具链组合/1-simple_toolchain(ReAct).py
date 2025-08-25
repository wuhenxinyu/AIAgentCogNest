"""
    基于 LangChain 的 Agent 框架（如 create_react_agent、AgentExecutor 等），通过 Agent 结合工具链，自动解析用户输入，智能选择和调用合适的工具，支持多轮推理和工具调用。
适合需要"智能决策+多工具自动调度"的复杂任务。
    通过组合三个工具来解决复杂任务，包括：
    • 文本分析工具 (TextAnalysisTool)：负责分析文本内容，提取统计信息和情感倾向。
    • 数据转换工具 (DataConversionTool)：实现不同数据格式之间的转换，目前支持JSON和CSV格式互转。
    • 文本处理工具 (TextProcessingTool)：提供文本处理功能，如统计行数、查找文本和替换文本。
 
    ReAct：是一种“思考-行动-观察”循环范式（Reasoning + Acting），即
        • 模型会先思考（Thought），
        • 再决定调用哪个工具（Action），
        • 然后观察工具的输出（Observation），
        • 再继续推理，直到得出结论。
"""
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain_community.llms import Tongyi
from langchain.memory import ConversationBufferMemory
import re
import json
from typing import List, Union, Dict, Any
import os
import dashscope

# 从环境变量获取 dashscope 的 API Key
os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here'
api_key = os.environ.get('DASHSCOPE_API_KEY')
dashscope.api_key = api_key

# 自定义工具1：文本分析工具

class TextAnalysisTool:
    """文本分析工具，用于分析文本内容"""
    
    def __init__(self):
        self.name = "文本分析"
        self.description = "分析文本内容，提取字数、字符数和情感倾向"
    
    def run(self, text: str) -> str:
        """分析文本内容
        
        参数:
            text: 要分析的文本
        返回:
            分析结果
        """
        # 简单的文本分析示例
        word_count = len(text.split())
        char_count = len(text)
        
        # 简单的情感分析（示例）
        positive_words = ["好", "优秀", "喜欢", "快乐", "成功", "美好"]
        negative_words = ["差", "糟糕", "讨厌", "悲伤", "失败", "痛苦"]
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        sentiment = "积极" if positive_count > negative_count else "消极" if negative_count > positive_count else "中性"
        
        return f"文本分析结果:\n- 字数: {word_count}\n- 字符数: {char_count}\n- 情感倾向: {sentiment}"

# 自定义工具2：数据转换工具
class DataConversionTool:
    """数据转换工具，用于在不同格式之间转换数据"""
    
    def __init__(self):
        self.name = "数据转换"
        self.description = "在不同数据格式之间转换，如JSON、CSV等"
    
    def run(self, input_data: str, input_format: str, output_format: str) -> str:
        """转换数据格式
        
        参数:
            input_data: 输入数据
            input_format: 输入格式
            output_format: 输出格式
        返回:
            转换后的数据
        """
        try:
            if input_format.lower() == "json" and output_format.lower() == "csv":
                # JSON到CSV的转换示例
                data = json.loads(input_data)
                if isinstance(data, list):
                    if not data:
                        return "空数据"
                    
                    # 获取所有可能的列
                    headers = set()
                    for item in data:
                        headers.update(item.keys())
                    headers = list(headers)
                    
                    # 创建CSV
                    csv = ",".join(headers) + "\n"
                    for item in data:
                        row = [str(item.get(header, "")) for header in headers]
                        csv += ",".join(row) + "\n"
                    
                    return csv
                else:
                    return "输入数据必须是JSON数组"
            
            elif input_format.lower() == "csv" and output_format.lower() == "json":
                # CSV到JSON的转换示例
                lines = input_data.strip().split("\n")
                if len(lines) < 2:
                    return "CSV数据至少需要标题行和数据行"
                
                headers = lines[0].split(",")
                result = []
                
                for line in lines[1:]:
                    values = line.split(",")
                    if len(values) != len(headers):
                        continue
                    
                    item = {}
                    for i, header in enumerate(headers):
                        item[header] = values[i]
                    result.append(item)
                
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            else:
                return f"不支持的转换: {input_format} -> {output_format}"
        
        except Exception as e:
            return f"转换失败: {str(e)}"

# 自定义工具3：文本处理工具
class TextProcessingTool:
    """文本处理工具，用于处理文本内容"""
    
    def __init__(self):
        self.name = "文本处理"
        self.description = "处理文本内容，如查找、替换、统计等"
    
    def run(self, operation: str, content: str, **kwargs) -> str:
        """处理文本内容
        
        参数:
            operation: 操作类型
            content: 文本内容
            **kwargs: 其他参数
        返回:
            处理结果
        """
        if operation == "count_lines":
            return f"文本共有 {len(content.splitlines())} 行"
        
        elif operation == "find_text":
            search_text = kwargs.get("search_text", "")
            if not search_text:
                return "请提供要查找的文本"
            
            lines = content.splitlines()
            matches = []
            
            for i, line in enumerate(lines):
                if search_text in line:
                    matches.append(f"第 {i+1} 行: {line}")
            
            if matches:
                return f"找到 {len(matches)} 处匹配:\n" + "\n".join(matches)
            else:
                return f"未找到文本 '{search_text}'"
        
        elif operation == "replace_text":
            old_text = kwargs.get("old_text", "")
            new_text = kwargs.get("new_text", "")
            
            if not old_text:
                return "请提供要替换的文本"
            
            new_content = content.replace(old_text, new_text)
            count = content.count(old_text)
            
            return f"替换完成，共替换 {count} 处。\n新内容:\n{new_content}"
        
        else:
            return f"不支持的操作: {operation}"

# 创建工具链,函数负责初始化各工具并组合成一个完整的工具链
def create_tool_chain():
    """创建工具链"""
    # 创建工具实例
    text_analysis = TextAnalysisTool()
    data_conversion = DataConversionTool()
    text_processing = TextProcessingTool()
    
    # 将工具包装为LangChain工具格式,工具注册与调用：每个工具需要以特定格式注册，包括名称、描述和执行函数
    tools = [
        Tool(
            name=text_analysis.name,
            func=text_analysis.run,
            description="分析文本内容，提取字数、字符数和情感倾向"
        ),
        Tool(
            name=data_conversion.name,
            func=data_conversion.run,
            description="在不同数据格式之间转换，如JSON、CSV等"
        ),
        Tool(
            name=text_processing.name,
            func=text_processing.run,
            description="处理文本内容，如查找、替换、统计等"
        )
    ]
    
    # 初始化语言模型
    llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=api_key)
    
    # 创建提示模板
    prompt = PromptTemplate.from_template(
        """你是一个有用的AI助手，可以使用以下工具:
{tools}
可用工具名称: {tool_names}

使用以下格式:
问题: 你需要回答的问题
思考: 你应该始终思考要做什么
行动: 要使用的工具名称，必须是 [{tool_names}] 中的一个
行动输入: 工具的输入
观察: 工具的结果
... (这个思考/行动/行动输入/观察可以重复 N 次)
思考: 我现在已经有了最终答案
回答: 对原始问题的最终回答

开始!
问题: {input}
思考: {agent_scratchpad}"""
    )
    
    """
    ReAct框架：使用LangChain的ReAct框架，允许代理执行思考-行动-观察的循环：
        1. **思考**：分析任务，确定下一步行动
        2. **行动**：调用适当的工具
        3. **观察**：分析工具返回的结果
        4. **重复**：根据观察结果继续思考，直到任务完成
    """
    # 创建Agent
    agent = create_react_agent(llm, tools, prompt)
    
    # 创建Agent执行器
    # 内存与上下文：系统使用 `ConversationBufferMemory` 存储对话历史，使代理能够参考之前的交互;memory=ConversationBufferMemory(memory_key="chat_history")
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=ConversationBufferMemory(memory_key="chat_history"),
        verbose=True,
        handle_parsing_errors=False  # 关闭自动重试, True会严格检查重试
    )
    
    return agent_executor

# 示例：使用工具链处理任务
def process_task(task_description):
    """
    使用工具链处理任务
    
    参数:
        task_description: 任务描述
    返回:
        处理结果
    """
    try:
        agent_executor = create_tool_chain()
        response = agent_executor.invoke({"input": task_description})
        return response["output"]  # 从返回的字典中提取输出
    except Exception as e:
        return f"处理任务时出错: {str(e)}"


"""
整体的工作流程是怎样的？
    Step1, 用户提交任务描述
    Step2, Agent分析任务，决定使用哪些工具
    Step3, Agent通过ReAct框架调用相应工具
    Step4, 系统整合各工具结果，生成最终回答
"""
# 示例用法
if __name__ == "__main__":
    # 示例1: 文本分析与处理
    task1 = "分析以下文本的情感倾向，并统计其中的行数：'这个产品非常好用，我很喜欢它的设计，使用体验非常棒！\n价格也很合理，推荐大家购买。\n客服态度也很好，解答问题很及时。'"
    print("任务1:", task1)
    print("结果:", process_task(task1))
    
    # 示例2: 数据格式转换
    task2 = "将以下CSV数据转换为JSON格式：'name,age,comment\n张三,25,这个产品很好\n李四,30,服务态度差\n王五,28,性价比高'"
    print("\n任务2:", task2)
    print("结果:", process_task(task2))