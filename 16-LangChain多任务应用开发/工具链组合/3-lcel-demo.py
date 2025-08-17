"""
   LCEL 是 LangChain 推出的链式表达式语言，支持用"|"操作符将各类单元（如Prompt、LLM、Parser等）组合。每个"|"左侧的输出会自动作为右侧的输入，实现数据流式传递。
    优势：
    • 代码简洁，逻辑清晰，易于多步任务编排。
    • 支持多分支、条件、并行等复杂链路。
    • 易于插拔、复用和调试每个子任务。
    典型用法：
    • 串联：`A | B | C`，A的输出传给B，B的输出传给C。
    • 分支：`{"x": A, "y": B}`，并行执行A和B。
    • 支持流式：如 `.stream()` 方法可边生成边消费。

    缺点：任务链的每一步由开发者显式指定（如先文本分析再统计行数，或条件分支）。不具备 Agent 的"自主决策"能力，所有流程和分支都需手动编排。
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Tongyi  # 导入通义千问Tongyi模型
from langchain_core.output_parsers import StrOutputParser
import dashscope
import os


# 从环境变量获取 dashscope 的 API Key
os.environ['DASHSCOPE_API_KEY'] = 'sk-58f051ae745e4bb19fdca31735105b11'
api_key = os.environ.get('DASHSCOPE_API_KEY')
dashscope.api_key = api_key

# stream=True 让LLM支持流式输出
llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=api_key, stream=True)


# 定义三个子任务：翻译->处理->回译
translate_to_en = ChatPromptTemplate.from_template("Translate this to English: {input}") | llm | StrOutputParser()
process_text = ChatPromptTemplate.from_template("Analyze this text: {text}") | llm | StrOutputParser()
translate_to_cn = ChatPromptTemplate.from_template("Translate this to Chinese: {output}") | llm | StrOutputParser()

# 组合成多任务链
# 定义一个工作流，该工作流将多个子任务按顺序组合起来。
# 1. 首先，使用字典 {"text": translate_to_en} 将输入传递给 translate_to_en 子任务，
#    该子任务会将输入文本翻译成英文。此步骤的输出会被包装在一个字典中，键为 "text"。
# 2. 接着，将上一步的输出传递给 process_text 子任务，该子任务会对英文文本进行分析。
# 3. 最后，将分析结果传递给 translate_to_cn 子任务，将分析后的英文文本翻译回中文。
# 通过 | 操作符将这些子任务按顺序连接起来，形成一个完整的工作流。
workflow = {"text": translate_to_en} | process_text | translate_to_cn
#workflow.invoke({"input": "北京有哪些好吃的地方，简略回答不超过200字"})

# 使用stream方法，边生成边打印
for chunk in workflow.stream({"input": "北京有哪些好吃的地方，简略回答不超过200字"}):
    print(chunk, end="", flush=True)
print()  # 换行