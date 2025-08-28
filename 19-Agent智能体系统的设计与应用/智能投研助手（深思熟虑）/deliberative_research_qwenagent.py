"""
深思熟虑智能体（Deliberative Agent）- 智能投研助手

基于Qwen-Agent实现的深思熟虑型智能体，适用于投资研究场景，能够整合数据，进行多步骤分析和推理，生成投资观点和研究报告。

核心流程：
    1. 感知：收集市场数据和信息
    2. 建模：构建内部世界模型，理解市场状态
    3. 推理：生成多个候选分析方案并模拟结果
    4. 决策：选择最优投资观点并形成报告
    5. 报告：生成完整研究报告
    
主要特点：
    1. 使用 @register_tool 装饰器定义了各个阶段的工具函数
    2. 采用会话隔离存储 _session_data 来保存不同用户的分析状态
    3. 提供了终端交互模式和Web图形界面模式
    4. 每个阶段的结果都会被存储，并传递给下一阶段
    5. 最终报告会保存到本地文件系统
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.tools.base import BaseTool, register_tool
from langchain_core.pydantic_v1 import BaseModel, Field

# 解决中文显示问题
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 配置DashScope API密钥
os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here'
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')
dashscope.timeout = 30

# 定义输出模型
class PerceptionOutput(BaseModel):
    """感知阶段输出的市场数据和信息"""
    market_overview: str = Field(..., description="市场概况和最新动态")
    key_indicators: Dict[str, str] = Field(..., description="关键经济和市场指标")
    recent_news: List[str] = Field(..., description="近期重要新闻")
    industry_trends: Dict[str, str] = Field(..., description="行业趋势分析")

class ModelingOutput(BaseModel):
    """建模阶段输出的内部世界模型"""
    market_state: str = Field(..., description="当前市场状态评估")
    economic_cycle: str = Field(..., description="经济周期判断")
    risk_factors: List[str] = Field(..., description="主要风险因素")
    opportunity_areas: List[str] = Field(..., description="潜在机会领域")
    market_sentiment: str = Field(..., description="市场情绪分析")

class ReasoningPlan(BaseModel):
    """推理阶段生成的候选分析方案"""
    plan_id: str = Field(..., description="方案ID")
    hypothesis: str = Field(..., description="投资假设")
    analysis_approach: str = Field(..., description="分析方法")
    expected_outcome: str = Field(..., description="预期结果")
    confidence_level: float = Field(..., description="置信度(0-1)")
    pros: List[str] = Field(..., description="方案优势")
    cons: List[str] = Field(..., description="方案劣势")

class DecisionOutput(BaseModel):
    """决策阶段选择的最优投资观点"""
    selected_plan_id: str = Field(..., description="选中的方案ID")
    investment_thesis: str = Field(..., description="投资论点")
    supporting_evidence: List[str] = Field(..., description="支持证据")
    risk_assessment: str = Field(..., description="风险评估")
    recommendation: str = Field(..., description="投资建议")
    timeframe: str = Field(..., description="时间框架")

# 会话隔离存储
_session_data = {}

def get_session_id(kwargs):
    """获取当前会话的唯一ID"""
    messages = kwargs.get('messages')
    if messages is not None:
        return id(messages)
    return 'default'

# 定义工具函数
@register_tool('perception')
class PerceptionTool(BaseTool):
    """感知阶段：收集和整理市场数据和信息"""
    description = '感知阶段：收集和整理市场数据和信息'
    parameters = [
        {
            'name': 'research_topic',
            'type': 'string',
            'description': '研究主题',
            'required': True
        },
        {
            'name': 'industry_focus',
            'type': 'string',
            'description': '行业焦点',
            'required': True
        },
        {
            'name': 'time_horizon',
            'type': 'string',
            'description': '时间范围(短期/中期/长期)',
            'required': True
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_community.llms import Tongyi
        from langchain_core.output_parsers import JsonOutputParser
        import json

        args = json.loads(params)
        research_topic = args['research_topic']
        industry_focus = args['industry_focus']
        time_horizon = args['time_horizon']

        # 准备提示
        PERCEPTION_PROMPT = """
        你是一个专业的投资研究分析师，请收集和整理关于以下研究主题的市场数据和信息：

        研究主题: {research_topic}
        行业焦点: {industry_focus}
        时间范围: {time_horizon}

        请从以下几个方面进行市场感知：
        1. 市场概况和最新动态
        2. 关键经济和市场指标
        3. 近期重要新闻（至少3条）
        4. 行业趋势分析（至少针对3个细分领域）

        根据你的专业知识和经验，提供尽可能详细和准确的信息。

        输出格式要求为JSON，包含以下字段：
        - market_overview: 字符串
        - key_indicators: 字典，键为指标名称，值为指标值和简要解释
        - recent_news: 字符串列表，每项为一条重要新闻
        - industry_trends: 字典，键为细分领域，值为趋势分析
        """

        prompt = ChatPromptTemplate.from_template(PERCEPTION_PROMPT)

        # 构建输入
        input_data = {
            "research_topic": research_topic,
            "industry_focus": industry_focus,
            "time_horizon": time_horizon
        }

        # 调用LLM
        llm = Tongyi(model_name="Qwen-Turbo-2025-04-28", dashscope_api_key=dashscope.api_key)
        chain = prompt | llm | JsonOutputParser()
        result = chain.invoke(input_data)

        # 存储结果到会话
        session_id = get_session_id(kwargs)
        if session_id not in _session_data:
            _session_data[session_id] = {}
        _session_data[session_id]['perception_data'] = result
        _session_data[session_id]['research_topic'] = research_topic
        _session_data[session_id]['industry_focus'] = industry_focus
        _session_data[session_id]['time_horizon'] = time_horizon

        return f"感知阶段完成，已收集市场数据和信息。\n{json.dumps(result, ensure_ascii=False, indent=2)}"

@register_tool('modeling')
class ModelingTool(BaseTool):
    """建模阶段：构建内部世界模型，理解市场状态"""
    description = '建模阶段：根据市场数据和信息构建内部世界模型'
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_community.llms import Tongyi
        from langchain_core.output_parsers import JsonOutputParser
        import json

        # 获取会话ID和感知数据
        session_id = get_session_id(kwargs)
        if session_id not in _session_data or 'perception_data' not in _session_data[session_id]:
            return "错误：缺少感知数据，请先调用perception工具。"

        perception_data = _session_data[session_id]['perception_data']
        research_topic = _session_data[session_id].get('research_topic', '未知')
        industry_focus = _session_data[session_id].get('industry_focus', '未知')
        time_horizon = _session_data[session_id].get('time_horizon', '未知')

        # 准备提示
        MODELING_PROMPT = """
        你是一个资深投资策略师，请根据以下市场数据和信息，构建市场内部模型，进行深度分析：

        研究主题: {research_topic}
        行业焦点: {industry_focus}
        时间范围: {time_horizon}

        市场数据和信息:
        {perception_data}

        请构建一个全面的市场内部模型，包括：
        1. 当前市场状态评估
        2. 经济周期判断
        3. 主要风险因素（至少3个）
        4. 潜在机会领域（至少3个）
        5. 市场情绪分析

        输出格式要求为JSON，包含以下字段：
        - market_state: 字符串
        - economic_cycle: 字符串
        - risk_factors: 字符串列表
        - opportunity_areas: 字符串列表
        - market_sentiment: 字符串
        """

        prompt = ChatPromptTemplate.from_template(MODELING_PROMPT)

        # 构建输入
        input_data = {
            "research_topic": research_topic,
            "industry_focus": industry_focus,
            "time_horizon": time_horizon,
            "perception_data": json.dumps(perception_data, ensure_ascii=False, indent=2)
        }

        # 调用LLM
        llm = Tongyi(model_name="Qwen-Turbo-2025-04-28", dashscope_api_key=dashscope.api_key)
        chain = prompt | llm | JsonOutputParser()
        result = chain.invoke(input_data)

        # 存储结果到会话
        _session_data[session_id]['world_model'] = result

        return f"建模阶段完成，已构建内部世界模型。\n{json.dumps(result, ensure_ascii=False, indent=2)}"

@register_tool('reasoning')
class ReasoningTool(BaseTool):
    """推理阶段：生成多个候选分析方案并模拟结果"""
    description = '推理阶段：根据市场模型生成候选分析方案'
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_community.llms import Tongyi
        from langchain_core.output_parsers import JsonOutputParser
        import json

        # 获取会话ID和世界模型
        session_id = get_session_id(kwargs)
        if session_id not in _session_data or 'world_model' not in _session_data[session_id]:
            return "错误：缺少世界模型，请先调用modeling工具。"

        world_model = _session_data[session_id]['world_model']
        research_topic = _session_data[session_id].get('research_topic', '未知')
        industry_focus = _session_data[session_id].get('industry_focus', '未知')
        time_horizon = _session_data[session_id].get('time_horizon', '未知')

        # 准备提示
        REASONING_PROMPT = """
        你是一个战略投资顾问，请根据以下市场模型，生成3个不同的投资分析方案：

        研究主题: {research_topic}
        行业焦点: {industry_focus}
        时间范围: {time_horizon}

        市场内部模型:
        {world_model}

        请为每个方案提供：
        1. 方案ID（简短标识符）
        2. 投资假设
        3. 分析方法
        4. 预期结果
        5. 置信度（0-1之间的小数）
        6. 方案优势（至少3点）
        7. 方案劣势（至少2点）

        这些方案应该有明显的差异，代表不同的投资思路或分析角度。

        输出格式要求为JSON数组，每个元素包含以下字段：
        - plan_id: 字符串
        - hypothesis: 字符串
        - analysis_approach: 字符串
        - expected_outcome: 字符串
        - confidence_level: 浮点数
        - pros: 字符串列表
        - cons: 字符串列表
        """

        prompt = ChatPromptTemplate.from_template(REASONING_PROMPT)

        # 构建输入
        input_data = {
            "research_topic": research_topic,
            "industry_focus": industry_focus,
            "time_horizon": time_horizon,
            "world_model": json.dumps(world_model, ensure_ascii=False, indent=2)
        }

        # 调用LLM
        llm = Tongyi(model_name="Qwen-Turbo-2025-04-28", dashscope_api_key=dashscope.api_key)
        chain = prompt | llm | JsonOutputParser()
        result = chain.invoke(input_data)

        # 存储结果到会话
        _session_data[session_id]['reasoning_plans'] = result

        return f"推理阶段完成，已生成候选分析方案。\n{json.dumps(result, ensure_ascii=False, indent=2)}"

@register_tool('decision')
class DecisionTool(BaseTool):
    """决策阶段：选择最优投资观点并形成报告"""
    description = '决策阶段：评估候选方案并选择最优投资观点'
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_community.llms import Tongyi
        from langchain_core.output_parsers import JsonOutputParser
        import json

        # 获取会话ID和候选方案
        session_id = get_session_id(kwargs)
        if session_id not in _session_data or 'reasoning_plans' not in _session_data[session_id]:
            return "错误：缺少候选方案，请先调用reasoning工具。"

        reasoning_plans = _session_data[session_id]['reasoning_plans']
        world_model = _session_data[session_id]['world_model']
        research_topic = _session_data[session_id].get('research_topic', '未知')
        industry_focus = _session_data[session_id].get('industry_focus', '未知')
        time_horizon = _session_data[session_id].get('time_horizon', '未知')

        # 准备提示
        DECISION_PROMPT = """
        你是一个投资决策委员会主席，请评估以下候选分析方案，选择最优方案并形成投资决策：

        研究主题: {research_topic}
        行业焦点: {industry_focus}
        时间范围: {time_horizon}

        市场内部模型:
        {world_model}

        候选分析方案:
        {reasoning_plans}

        请基于方案的假设、分析方法、预期结果、置信度以及优缺点，选择最优的投资方案，并给出详细的决策理由。
        你的决策应该综合考虑投资潜力、风险水平和时间框架的匹配度。

        输出格式要求为JSON，包含以下字段：
        - selected_plan_id: 字符串
        - investment_thesis: 字符串
        - supporting_evidence: 字符串列表
        - risk_assessment: 字符串
        - recommendation: 字符串
        - timeframe: 字符串
        """

        prompt = ChatPromptTemplate.from_template(DECISION_PROMPT)

        # 构建输入
        input_data = {
            "research_topic": research_topic,
            "industry_focus": industry_focus,
            "time_horizon": time_horizon,
            "world_model": json.dumps(world_model, ensure_ascii=False, indent=2),
            "reasoning_plans": json.dumps(reasoning_plans, ensure_ascii=False, indent=2)
        }

        # 调用LLM
        llm = Tongyi(model_name="Qwen-Turbo-2025-04-28", dashscope_api_key=dashscope.api_key)
        chain = prompt | llm | JsonOutputParser()
        result = chain.invoke(input_data)

        # 存储结果到会话
        _session_data[session_id]['selected_plan'] = result

        return f"决策阶段完成，已选择最优投资观点。\n{json.dumps(result, ensure_ascii=False, indent=2)}"

@register_tool('report')
class ReportTool(BaseTool):
    """报告阶段：生成完整研究报告"""
    description = '报告阶段：根据分析结果生成完整研究报告'
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_community.llms import Tongyi
        from langchain_core.output_parsers import StrOutputParser
        import json
        import os

        # 获取会话ID和选定方案
        session_id = get_session_id(kwargs)
        if session_id not in _session_data or 'selected_plan' not in _session_data[session_id]:
            return "错误：缺少选定方案，请先调用decision工具。"

        selected_plan = _session_data[session_id]['selected_plan']
        perception_data = _session_data[session_id]['perception_data']
        world_model = _session_data[session_id]['world_model']
        research_topic = _session_data[session_id].get('research_topic', '未知')
        industry_focus = _session_data[session_id].get('industry_focus', '未知')
        time_horizon = _session_data[session_id].get('time_horizon', '未知')

        # 准备提示
        REPORT_PROMPT = """
        你是一个专业的投资研究报告撰写人，请根据以下信息生成一份完整的投资研究报告：

        研究主题: {research_topic}
        行业焦点: {industry_focus}
        时间范围: {time_horizon}

        市场数据和信息:
        {perception_data}

        市场内部模型:
        {world_model}

        选定的投资决策:
        {selected_plan}

        请生成一份结构完整、逻辑清晰的投研报告，包括但不限于：
        1. 报告标题和摘要
        2. 市场和行业背景
        3. 核心投资观点
        4. 详细分析论证
        5. 风险因素
        6. 投资建议
        7. 时间框架和预期回报

        报告应当专业、客观，同时提供足够的分析深度和洞见。
        """

        prompt = ChatPromptTemplate.from_template(REPORT_PROMPT)

        # 构建输入
        input_data = {
            "research_topic": research_topic,
            "industry_focus": industry_focus,
            "time_horizon": time_horizon,
            "perception_data": json.dumps(perception_data, ensure_ascii=False, indent=2),
            "world_model": json.dumps(world_model, ensure_ascii=False, indent=2),
            "selected_plan": json.dumps(selected_plan, ensure_ascii=False, indent=2)
        }

        # 调用LLM
        llm = Tongyi(model_name="Qwen-Turbo-2025-04-28", dashscope_api_key=dashscope.api_key)
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke(input_data)

        # 存储结果到会话
        _session_data[session_id]['final_report'] = result

        # 保存报告到文件
        report_dir = os.path.join(os.path.dirname(__file__), 'reports')
        os.makedirs(report_dir, exist_ok=True)
        report_filename = f"{research_topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path = os.path.join(report_dir, report_filename)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(result)

        return f"报告生成完成，已保存至 {report_path}\n\n{result}"

# 系统提示词
SYSTEM_PROMPT = """
你是一个深思熟虑的智能投研助手，专注于为用户提供专业、全面的投资研究分析和报告。

你的核心工作流程包括五个阶段：
1. 感知（perception）：收集市场数据和信息
2. 建模（modeling）：构建内部世界模型，理解市场状态
3. 推理（reasoning）：生成多个候选分析方案并模拟结果
4. 决策（decision）：选择最优投资观点并形成报告
5. 报告（report）：生成完整研究报告

当用户提出投资研究需求时，你需要：
1. 首先询问用户研究主题、行业焦点和时间范围
2. 然后依次调用perception、modeling、reasoning、decision和report工具完成分析
3. 每个阶段完成后，向用户展示该阶段的结果，并询问是否继续
4. 最终生成完整的研究报告并提供给用户

请严格按照工作流程执行，确保分析的全面性和深度。
"""

# 初始化智能投研助手服务
def init_research_agent_service():
    """初始化智能投研助手服务"""
    llm_cfg = {
        'model': 'qwen-turbo-2025-04-28',
        'timeout': 60,
        'retry_count': 3,
    }
    try:
        bot = Assistant(
            llm=llm_cfg,
            name='深思熟虑智能投研助手',
            description='专业的投资研究分析助手，能够进行多步骤分析和推理，生成投资观点和研究报告。',
            system_message=SYSTEM_PROMPT,
            function_list=['perception', 'modeling', 'reasoning', 'decision', 'report'],
        )
        print("智能投研助手初始化成功！")
        return bot
    except Exception as e:
        print(f"智能投研助手初始化失败: {str(e)}")
        raise

# 终端交互模式
def app_tui():
    """终端交互模式"""
    try:
        # 初始化助手
        bot = init_research_agent_service()

        # 对话历史
        messages = []
        print("欢迎使用深思熟虑智能投研助手！请输入您的研究需求。")
        while True:
            try:
                # 获取用户输入
                query = input('user question: ')

                # 输入验证
                if not query:
                    print('user question cannot be empty！')
                    continue
                     
                # 构建消息
                messages.append({'role': 'user', 'content': query})

                print("正在处理您的请求...")
                # 运行助手并处理响应
                response = []
                for response in bot.run(messages):
                    print('bot response:', response)
                messages.extend(response)
            except Exception as e:
                print(f"处理请求时出错: {str(e)}")
                print("请重试或输入新的问题")
    except Exception as e:
        print(f"启动终端模式失败: {str(e)}")

# 图形界面模式
def app_gui():
    """图形界面模式，提供Web图形界面"""
    try:
        print("正在启动Web界面...")
        # 初始化助手
        bot = init_research_agent_service()
        # 配置聊天界面，列举3个典型研究问题
        chatbot_config = {
            'prompt.suggestions': [
                '分析2025年中国新能源汽车行业的投资机会',
                '研究人工智能在金融领域的应用前景',
                '评估2025年全球半导体行业的市场趋势'
            ]
        }
        print("Web界面准备就绪，正在启动服务...")
        # 启动Web界面
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
    except Exception as e:
        print(f"启动Web界面失败: {str(e)}")
        print("请检查网络连接和API Key配置")

if __name__ == '__main__':
    # 运行模式选择
    app_gui()  # 图形界面模式（默认）
    # app_tui()  # 终端交互模式