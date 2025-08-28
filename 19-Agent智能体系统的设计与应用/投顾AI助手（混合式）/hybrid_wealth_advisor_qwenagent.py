"""
混合智能体（Hybrid Agent）- 财富管理投顾AI助手（qwen-agent实现版）

基于qwen-agent实现的混合型智能体，结合反应式架构的即时响应能力和深思熟虑架构的长期规划能力，
通过协调层动态切换处理模式，提供智能化财富管理咨询服务。

三层架构：
    1. 底层（反应式）：即时响应客户查询，提供快速反馈
    2. 中层（协调）：评估任务类型和优先级，动态选择处理模式
    3. 顶层（深思熟虑）：进行复杂的投资分析和长期财务规划

运作机制：通过qwen-agent的工具调用机制动态切换模式：
    ▸ 紧急情况 → 启用反应式快速响应
    ▸ 常规情况 → 启动深思熟虑规划
    ▸ 特殊需求 → 自定义模式（如风险分析、资产配置建议等）

核心优势：
    • 兼具实时响应能力（毫秒级）
    • 保留战略规划优势（长期目标）
    • 高度可定制化（支持客户特定需求）

使用方法：
1.运行程序后可以选择终端交互模式或Web图形界面模式
2.选择客户类型（平衡型或进取型）
3.输入您的投资问题，系统会自动评估并选择合适的处理模式
4.查看助手的响应结果
5.可以继续输入问题，系统会根据上下文进行智能回复
6.退出程序
"""

# 导入必要的库
import os
import json
import asyncio
import time
import threading
from typing import Dict, List, Any, Literal, Optional
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.tools.base import BaseTool, register_tool
from datetime import datetime


"""
1.添加RateLimiter类 ：这是一个简单的请求频率控制器，用于限制每秒钟的最大请求次数
    - max_calls=3 ：设置每秒钟最多发出3次请求
    - period=1 ：设置时间周期为1秒
    - wait() 方法：在每次请求前调用，如果已达到最大调用次数则等待
2.创建全局频率控制器实例 ： rate_limiter = RateLimiter(max_calls=3, period=1)
3.在各工具函数中应用频率控制 ：在每个工具的 call() 方法开始处添加 rate_limiter.wait() 调用

### 工作原理
    1、频率控制器会跟踪最近1秒内的所有请求时间
    2、当收到新请求时，会检查最近1秒内的请求数量
    3、如果已达到或超过3次，控制器会计算需要等待的时间并暂停执行
    4、等待时间结束后，才会继续处理请求并记录新的请求时间
"""
# 请求频率限制器
# 修改RateLimiter类，增加更严格的频率控制和指数退避重试机制
class RateLimiter:
    def __init__(self, max_calls=1, period=1.5):  # 降低调用频率，改为每秒最多1次
        """初始化频率限制器
        max_calls: 每period秒最多调用次数
        period: 时间周期(秒)
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = threading.Lock()
        
    def wait(self):
        """等待直到可以进行下一次调用"""
        with self.lock:
            current_time = time.time()
            
            # 移除过期的调用记录
            self.calls = [t for t in self.calls if current_time - t < self.period]
            
            # 如果已达到最大调用次数，则等待
            if len(self.calls) >= self.max_calls:
                wait_time = self.period - (current_time - self.calls[0])
                # 增加额外的安全等待时间，避免接近限制
                safe_wait_time = wait_time + 0.2  # 额外等待0.2秒
                if safe_wait_time > 0:
                    time.sleep(safe_wait_time)
                
                # 再次检查并移除过期的调用记录
                current_time = time.time()
                self.calls = [t for t in self.calls if current_time - t < self.period]
            
            # 记录新的调用时间
            self.calls.append(current_time)

# 创建全局的请求频率限制器(更严格的设置)
rate_limiter = RateLimiter(max_calls=1, period=1.5)

# 设置API密钥
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', 'your_api_key_here')
dashscope.timeout = 30  # 设置超时时间为30秒

# 示例客户画像数据
SAMPLE_CUSTOMER_PROFILES = {
    "customer1": {
        "customer_id": "C10012345",
        "risk_tolerance": "平衡型",
        "investment_horizon": "中期",
        "financial_goals": ["退休规划", "子女教育金"],
        "investment_preferences": ["ESG投资", "科技行业"],
        "portfolio_value": 1500000.0,
        "current_allocations": {
            "股票": 0.40,
            "债券": 0.30,
            "现金": 0.10,
            "另类投资": 0.20
        }
    },
    "customer2": {
        "customer_id": "C10067890",
        "risk_tolerance": "进取型",
        "investment_horizon": "长期",
        "financial_goals": ["财富增长", "资产配置多元化"],
        "investment_preferences": ["新兴市场", "高成长行业"],
        "portfolio_value": 3000000.0,
        "current_allocations": {
            "股票": 0.65,
            "债券": 0.15,
            "现金": 0.05,
            "另类投资": 0.15
        }
    }
}

# ====== 查询类型评估工具 ======
@register_tool('assess_query_type')
class AssessQueryTypeTool(BaseTool):
    """
    评估用户查询类型的工具，用于确定采用何种处理模式
    """
    description = '评估用户查询的类型和复杂度，确定应采用的处理模式'
    parameters = [{
        'name': 'user_query',
        'type': 'string',
        'description': '用户的查询内容',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        user_query = args['user_query']
        
        # 调用频率限制器
        rate_limiter.wait()
        
        # 准备评估提示
        assessment_prompt = f"""你是一个财富管理投顾AI助手的协调层。请评估以下用户查询，确定其类型和应该采用的处理模式。

用户查询: {user_query}

请判断:
1. 查询类型: 
   - "emergency": 紧急的或直接的查询，需要立即响应（如市场状况、账户信息、产品信息等）
   - "informational": 信息性的查询，需要特定领域知识（如税务政策、投资工具介绍等）
   - "analytical": 需要深度分析的查询（如投资组合优化、长期理财规划等）

2. 建议的处理模式:
   - "reactive": 适用于需要快速反应的查询
   - "deliberative": 适用于需要深度思考和分析的查询

请以JSON格式返回结果，包含以下字段:
- query_type: 查询类型（上述三种类型之一）
- processing_mode: 处理模式（上述两种模式之一）
- reasoning: 决策理由的简要说明
"""
        
        # 创建临时LLM实例进行评估
        from langchain_community.llms import Tongyi
        llm = Tongyi(model_name="Qwen-Turbo-2025-04-28", dashscope_api_key=dashscope.api_key)
        result = llm.invoke(assessment_prompt)
        
        return result

# ====== 反应式处理工具 ======
@register_tool('reactive_processing')
class ReactiveProcessingTool(BaseTool):
    """
    反应式处理工具，用于快速响应用户的简单查询
    """
    description = '用于快速响应用户的简单查询，提供直接的回答'
    parameters = [{
        'name': 'user_query',
        'type': 'string',
        'description': '用户的查询内容',
        'required': True
    }, {
        'name': 'customer_profile',
        'type': 'string',
        'description': '客户画像信息的JSON字符串',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        user_query = args['user_query']
        customer_profile = args['customer_profile']
        
        # 调用频率限制器
        rate_limiter.wait()
        
        # 准备反应式处理提示
        reactive_prompt = f"""你是一个财富管理投顾AI助手，专注于提供快速准确的响应。请针对用户的查询提供直接的回答。

用户查询: {user_query}

客户信息:
{customer_profile}

请提供直接、简洁的回答，避免过多的解释和分析。
"""
        
        # 创建临时LLM实例进行处理
        from langchain_community.llms import Tongyi
        llm = Tongyi(model_name="Qwen-Turbo-2025-04-28", dashscope_api_key=dashscope.api_key)
        result = llm.invoke(reactive_prompt)
        
        return result

# ====== 深思熟虑处理工具 - 数据收集阶段 ======
@register_tool('data_collection')
class DataCollectionTool(BaseTool):
    """
    数据收集工具，用于为用户的复杂查询收集相关的市场和财务数据
    """
    description = '用于为用户的复杂查询收集相关的市场和财务数据'
    parameters = [{
        'name': 'user_query',
        'type': 'string',
        'description': '用户的查询内容',
        'required': True
    }, {
        'name': 'customer_profile',
        'type': 'string',
        'description': '客户画像信息的JSON字符串',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        user_query = args['user_query']
        customer_profile = args['customer_profile']
        
        data_collection_prompt = f"""你是一个财富管理投顾AI助手的数据收集模块。基于以下用户查询，收集相关的市场和财务数据。

用户查询: {user_query}

客户信息:
{customer_profile}

请确定需要收集的数据，并生成合理的模拟数据，以JSON格式返回。
"""
        
        # 调用频率限制器
        rate_limiter.wait()
        # 创建临时LLM实例进行数据收集
        from langchain_community.llms import Tongyi
        llm = Tongyi(model_name="Qwen-Turbo-2025-04-28", dashscope_api_key=dashscope.api_key)
        market_data = llm.invoke(data_collection_prompt)
        
        return market_data

# ====== 深思熟虑处理工具 - 深度分析阶段 ======
@register_tool('deep_analysis')
class DeepAnalysisTool(BaseTool):
    """
    深度分析工具，用于对用户的投资情况进行深入分析
    """
    description = '用于对用户的投资情况进行深入分析'
    parameters = [{
        'name': 'user_query',
        'type': 'string',
        'description': '用户的查询内容',
        'required': True
    }, {
        'name': 'customer_profile',
        'type': 'string',
        'description': '客户画像信息的JSON字符串',
        'required': True
    }, {
        'name': 'market_data',
        'type': 'string',
        'description': '市场数据的JSON字符串',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        user_query = args['user_query']
        customer_profile = args['customer_profile']
        market_data = args['market_data']
        
        analysis_prompt = f"""你是一个财富管理投顾AI助手的分析引擎。请根据收集的数据对用户的投资情况进行深入分析。

用户查询: {user_query}

客户信息:
{customer_profile}

市场数据:
{market_data}

请提供全面的投资分析，包括:
1. 当前市场状况评估
2. 客户投资组合分析
3. 个性化投资建议
4. 风险评估
5. 预期结果和回报预测
"""
        
        # 调用频率限制器
        rate_limiter.wait()
        from langchain_community.llms import Tongyi
        llm = Tongyi(model_name="Qwen-Turbo-2025-04-28", dashscope_api_key=dashscope.api_key)
        analysis_results = llm.invoke(analysis_prompt)
        
        return analysis_results

# 例如在RecommendationGenerationTool类中：
@register_tool('recommendation_generation')
class RecommendationGenerationTool(BaseTool):
    """
    建议生成工具，用于根据深入分析结果为客户准备最终的咨询建议
    """
    description = '用于根据深入分析结果为客户准备最终的咨询建议'
    parameters = [{
        'name': 'user_query',
        'type': 'string',
        'description': '用户的查询内容',
        'required': True
    }, {
        'name': 'customer_profile',
        'type': 'string',
        'description': '客户画像信息的JSON字符串',
        'required': True
    }, {
        'name': 'analysis_results',
        'type': 'string',
        'description': '深度分析结果',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        try:
            # 健壮的参数解析，处理不同格式的输入
            if isinstance(params, str):
                try:
                    args = json.loads(params)
                except json.JSONDecodeError:
                    # 如果JSON解析失败，尝试将params作为整个analysis_results参数
                    # 创建一个默认的参数字典
                    args = {
                        'user_query': '用户投资咨询',
                        'customer_profile': json.dumps(SAMPLE_CUSTOMER_PROFILES['customer1'], ensure_ascii=False),
                        'analysis_results': params
                    }
            elif isinstance(params, dict):
                args = params
            else:
                # 处理其他可能的类型
                args = {
                    'user_query': '用户投资咨询',
                    'customer_profile': json.dumps(SAMPLE_CUSTOMER_PROFILES['customer1'], ensure_ascii=False),
                    'analysis_results': str(params)
                }
            
            # 确保所有必需参数都存在
            user_query = args.get('user_query', '用户投资咨询')
            customer_profile = args.get('customer_profile', json.dumps(SAMPLE_CUSTOMER_PROFILES['customer1'], ensure_ascii=False))
            analysis_results = args.get('analysis_results', '')
            
            recommendation_prompt = f"""你是一个财富管理投顾AI助手。请根据深入分析结果，为客户准备最终的咨询建议。

用户查询: {user_query}

客户信息:
{customer_profile}

分析结果:
{analysis_results}

请提供专业、个性化且详细的投资建议，语言应友好易懂，避免过多专业术语。建议应包括:
1. 总体投资策略
2. 具体行动步骤
3. 资产配置建议
4. 风险管理策略
5. 时间框架
6. 预期收益
7. 后续跟进计划
"""
            
            # 调用频率限制器
            rate_limiter.wait()
            from langchain_community.llms import Tongyi
            llm = Tongyi(model_name="Qwen-Turbo-2025-04-28", dashscope_api_key=dashscope.api_key)
            final_recommendation = llm.invoke(recommendation_prompt)
            
            return final_recommendation
        except Exception as e:
            # 捕获所有异常并返回友好的错误消息
            return f"生成建议时发生错误: {str(e)}"

# ====== 实时市场数据查询工具 ======
@register_tool('query_market_data')
class QueryMarketDataTool(BaseTool):
    """
    实时市场数据查询工具，提供最新的市场行情信息
    """
    description = '用于查询实时市场数据，如指数、股票等的最新行情'
    parameters = [{
        'name': 'market_type',
        'type': 'string',
        'description': '市场类型，如"上证指数"、"深证成指"等',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        market_type = args['market_type']
        
        # 模拟市场数据，实际应用中可接入真实的市场数据API
        if market_type == "上证指数":
            return "上证指数 当前点位: 3125.62，涨跌: 6.32，涨跌幅: 0.20%（模拟数据）"
        elif market_type == "深证成指":
            return "深证成指 当前点位: 10253.85，涨跌: 28.64，涨跌幅: 0.28%（模拟数据）"
        elif market_type == "创业板指":
            return "创业板指 当前点位: 2052.36，涨跌: 15.78，涨跌幅: 0.77%（模拟数据）"
        else:
            return f"{market_type} 当前没有可用的模拟数据"

# ====== 客户画像查询工具 ======
@register_tool('get_customer_profile')
class GetCustomerProfileTool(BaseTool):
    """
    客户画像查询工具，提供客户的详细信息
    """
    description = '用于查询客户的详细画像信息，包括风险承受能力、投资期限、财务目标等'
    parameters = [{
        'name': 'customer_id',
        'type': 'string',
        'description': '客户ID，如"customer1"、"customer2"等',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        customer_id = args['customer_id']
        
        if customer_id in SAMPLE_CUSTOMER_PROFILES:
            return json.dumps(SAMPLE_CUSTOMER_PROFILES[customer_id], ensure_ascii=False, indent=2)
        else:
            return f"未找到客户ID为{customer_id}的客户画像信息"

# ====== 系统提示和函数描述 ======
# 修改前的系统提示
# system_prompt = """我是一个财富管理投顾AI助手，能够为客户提供专业的投资建议和财务规划服务。

# 我的主要功能包括：
# 1. 评估用户查询类型，确定合适的处理模式
# 2. 提供快速响应（反应式处理）：适用于简单、紧急的查询
# 3. 提供深度分析（深思熟虑处理）：适用于复杂、需要深入研究的投资问题
# 4. 查询实时市场数据
# 5. 获取客户画像信息

# 我会根据客户的风险承受能力、投资期限、财务目标等因素，提供个性化的投资建议。
# """

# 修改后的系统提示，添加了JSON格式示例
system_prompt = """我是一个财富管理投顾AI助手，能够为客户提供专业的投资建议和财务规划服务。

我的主要功能包括：
1. 评估用户查询类型，确定合适的处理模式
2. 提供快速响应（反应式处理）：适用于简单、紧急的查询
3. 提供深度分析（深思熟虑处理）：适用于复杂、需要深入研究的投资问题
   - 对于需要深度分析的问题，请严格按照以下流程执行：
     a. 首先调用data_collection工具收集相关数据
     b. 然后调用deep_analysis工具进行深入分析
     c. 最后必须调用recommendation_generation工具生成最终建议
4. 查询实时市场数据
5. 获取客户画像信息

我会根据客户的风险承受能力、投资期限、财务目标等因素，提供个性化的投资建议。

请严格按照以下JSON格式调用工具：
{
  "name": "工具名称",
  "parameters": {
    "参数名1": "参数值1",
    "参数名2": "参数值2"
  }
}

例如，调用评估查询类型工具的格式应为：
{
  "name": "assess_query_type",
  "parameters": {
    "user_query": "用户的具体问题"
  }
}

调用深度分析工具的格式应为：
{
  "name": "deep_analysis",
  "parameters": {
    "user_query": "用户的具体问题",
    "customer_profile": "客户画像信息的JSON字符串",
    "market_data": "市场数据的JSON字符串"
  }
}

调用建议生成工具的格式应为：
{
  "name": "recommendation_generation",
  "parameters": {
    "user_query": "用户的具体问题",
    "customer_profile": "客户画像信息的JSON字符串",
    "analysis_results": "深度分析结果"
  }
}
"""

# 定义函数描述列表，用于告知大模型可调用的函数及其参数信息
functions_desc = [
    {
        "name": "assess_query_type",
        "description": "评估用户查询的类型和复杂度，确定应采用的处理模式",
        "parameters": {
            "type": "object",
            "properties": {
                "user_query": {
                    "type": "string",
                    "description": "用户的查询内容"
                }
            },
            "required": ["user_query"]
        }
    },
    {
        "name": "reactive_processing",
        "description": "用于快速响应用户的简单查询，提供直接的回答",
        "parameters": {
            "type": "object",
            "properties": {
                "user_query": {
                    "type": "string",
                    "description": "用户的查询内容"
                },
                "customer_profile": {
                    "type": "string",
                    "description": "客户画像信息的JSON字符串"
                }
            },
            "required": ["user_query", "customer_profile"]
        }
    },
    {
        "name": "data_collection",
        "description": "用于为用户的复杂查询收集相关的市场和财务数据",
        "parameters": {
            "type": "object",
            "properties": {
                "user_query": {
                    "type": "string",
                    "description": "用户的查询内容"
                },
                "customer_profile": {
                    "type": "string",
                    "description": "客户画像信息的JSON字符串"
                }
            },
            "required": ["user_query", "customer_profile"]
        }
    },
    {
        "name": "deep_analysis",
        "description": "用于对用户的投资情况进行深入分析",
        "parameters": {
            "type": "object",
            "properties": {
                "user_query": {
                    "type": "string",
                    "description": "用户的查询内容"
                },
                "customer_profile": {
                    "type": "string",
                    "description": "客户画像信息的JSON字符串"
                },
                "market_data": {
                    "type": "string",
                    "description": "市场数据的JSON字符串"
                }
            },
            "required": ["user_query", "customer_profile", "market_data"]
        }
    },
    {
        "name": "recommendation_generation",
        "description": "用于根据深入分析结果为客户准备最终的咨询建议",
        "parameters": {
            "type": "object",
            "properties": {
                "user_query": {
                    "type": "string",
                    "description": "用户的查询内容"
                },
                "customer_profile": {
                    "type": "string",
                    "description": "客户画像信息的JSON字符串"
                },
                "analysis_results": {
                    "type": "string",
                    "description": "深度分析结果"
                }
            },
            "required": ["user_query", "customer_profile", "analysis_results"]
        }
    },
    {
        "name": "query_market_data",
        "description": "用于查询实时市场数据，如指数、股票等的最新行情",
        "parameters": {
            "type": "object",
            "properties": {
                "market_type": {
                    "type": "string",
                    "description": "市场类型，如\"上证指数\"、\"深证成指\"等"
                }
            },
            "required": ["market_type"]
        }
    },
    {
        "name": "get_customer_profile",
        "description": "用于查询客户的详细画像信息，包括风险承受能力、投资期限、财务目标等",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "客户ID，如\"customer1\"、\"customer2\"等"
                }
            },
            "required": ["customer_id"]
        }
    }
]

# ====== 初始化投顾助手服务 ======
def init_agent_service():
    """初始化财富管理投顾助手服务"""
    llm_cfg = {
        'model': 'qwen-turbo-2025-04-28',
        'timeout': 90,  # 增加超时时间到60秒
        'retry_count': 3,  # 增加重试次数到5次
        'retry_if': lambda e: 'Throttling.RateQuota' in str(e),  # 专门针对限流错误进行重试
    }
    try:
        # 调用频率限制器
        rate_limiter.wait()
        
        bot = Assistant(
            llm=llm_cfg,
            name='财富管理投顾AI助手',
            description='专业的财富管理和投资咨询服务',
            system_message=system_prompt,
            function_list=functions_desc,
        )
        print("助手初始化成功！")
        return bot
    except Exception as e:
        print(f"助手初始化失败: {str(e)}")
        raise

# ====== 终端交互模式 ======
def app_tui():
    """终端交互模式
    
    提供命令行交互界面，支持：
    - 连续对话
    - 实时响应
    """
    try:
        # 初始化助手
        bot = init_agent_service()

        # 对话历史
        messages = []
        # 选择客户
        print("请选择客户类型:")
        print("1. 平衡型投资者")
        print("2. 进取型投资者")
        customer_choice = input("请输入选项(1-2): ")
        customer_id = "customer1" if customer_choice != "2" else "customer2"
        customer_profile = json.dumps(SAMPLE_CUSTOMER_PROFILES[customer_id], ensure_ascii=False)
        print(f"已选择客户: {SAMPLE_CUSTOMER_PROFILES[customer_id]['risk_tolerance']}投资者\n")
        
        print("=== 混合智能体 - 财富管理投顾AI助手 ===")
        print("输入'退出'或'quit'结束对话\n")
        
        while True:
            try:
                # 获取用户输入
                query = input('请输入您的问题: ')
                if query.lower() in ['退出', 'quit', 'exit']:
                    print("感谢使用财富管理投顾AI助手，再见！")
                    break
                
                # 输入验证
                if not query:
                    print('问题不能为空，请重新输入！')
                    continue
                    
                # 构建消息，添加客户信息到用户查询中
                enriched_query = f"""用户问题: {query}

客户信息:
{customer_profile}
"""
                
                messages.append({'role': 'user', 'content': enriched_query})

                print("正在处理您的请求...")
                start_time = datetime.now()
                
                # 运行助手并处理响应
                response = []
                for res in bot.run(messages):
                    response.extend(res)
                    # 显示助手的回复
                    for msg in res:
                        if msg.get('role') == 'assistant':
                            print(f"助手回复: {msg.get('content')}")
                
                messages.extend(response)
                
                end_time = datetime.now()
                process_time = (end_time - start_time).total_seconds()
                print(f"处理用时: {process_time:.2f}秒\n")
                
            except Exception as e:
                print(f"处理请求时出错: {str(e)}")
                print("请重试或输入新的问题\n")
    except Exception as e:
        print(f"启动终端模式失败: {str(e)}")

# ====== 图形界面模式 ======
def app_gui():
    """图形界面模式，提供Web图形界面"""
    try:
        print("正在启动Web界面...")
        # 初始化助手
        bot = init_agent_service()
        # 配置聊天界面，列举3个典型投资问题
        chatbot_config = {
            'prompt.suggestions': [
                '今天上证指数的表现如何？',
                '我的投资组合中科技股占比是多少？',
                '根据当前市场情况，我应该如何调整投资组合以应对可能的经济衰退？'
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

# ====== 主函数 ======
if __name__ == '__main__':
    # 运行模式选择
    print("请选择运行模式:")
    print("1. 终端交互模式")
    print("2. Web图形界面模式")
    mode_choice = input("请输入选项(1-2): ")
    
    if mode_choice == "2":
        app_gui()  # 图形界面模式
    else:
        app_tui()  # 终端交互模式（默认）
