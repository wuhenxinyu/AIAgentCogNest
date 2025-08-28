"""
私募基金运作指引问答助手 - 反应式智能体实现

适合反应式架构的私募基金问答助手，使用Agent模式实现主动思考和工具选择。

采用反应式智能体架构设计，区别于传统的工作流模式，该智能体能够主动思考、自主决策并选择合适的工具来回答用户问题，具备真正的智能体特性。
    1. 自主决策：智能体能够根据问题类型自主选择使用哪种工具回答问题
    2. 透明思考过程：展示清晰的思考过程，使决策链路可追踪
    3. 知识边界感知：明确区分知识库内容和模型知识
    4. 多工具协作：集成多种工具，包括关键词搜索、类别查询和直接问答
    5. 异常处理：妥善处理边界情况和超出知识范围的问题，避免错误回答
基于LangChain的Agent框架实现，使用通义千问(Qwen-Turbo)模型作为底层大语言模型，通过工具选择、思考链路和自主决策能力，提供高质量的私募基金规则咨询服务。

注意： TONGYI和OPENAI的模型参数有差异，需要注意模型参数的设置。否则会出现NoneType：
   # Tongyi 的响应格式通常更稳定
    response = {
        "output": {
            "text": "具体回答内容"
        }
    }

    # OpenAI 兼容API的响应可能多样化
    response = {
        "choices": [{"text": "内容"}],  # 旧格式
        "choices": [{"message": {"content": "内容"}}],  # 新格式
        "completion": "内容"  # 其他格式
    }
    错误处理机制不同：Tongyi 类可能有更完善的错误处理机制，能够处理模型返回的异常情况，而 OpenAI 类对非标准API的兼容性较差，需要额外的处理逻辑。
"""

import re
from typing import List, Dict, Any, Union
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain_community.llms import Tongyi
from langchain.chains import LLMChain
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import PromptTemplate
from langchain.llms.base import BaseLLM
import os

# 通义千问API密钥
os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here'
# 获取环境变量中的 DASHSCOPE_API_KEY
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')

# Step1. 数据准备系统基于私募基金运作规则构建了精简的知识库(FUND_RULES_DB)，每条规则包含ID、分类、问题和答案
# 简化的私募基金规则数据库
FUND_RULES_DB = [
    {
        "id": "rule001",
        "category": "设立与募集",
        "question": "私募基金的合格投资者标准是什么？",
        "answer": "合格投资者是指具备相应风险识别能力和风险承担能力，投资于单只私募基金的金额不低于100万元且符合下列条件之一的单位和个人：\n1. 净资产不低于1000万元的单位\n2. 金融资产不低于300万元或者最近三年个人年均收入不低于50万元的个人"
    },
    {
        "id": "rule002",
        "category": "设立与募集",
        "question": "私募基金的最低募集规模要求是多少？",
        "answer": "私募证券投资基金的最低募集规模不得低于人民币1000万元。对于私募股权基金、创业投资基金等其他类型的私募基金，监管规定更加灵活，通常需符合基金合同的约定。"
    },
    {
        "id": "rule014",
        "category": "监管规定",
        "question": "私募基金管理人的风险准备金要求是什么？",
        "answer": "私募证券基金管理人应当按照管理费收入的10%计提风险准备金，主要用于赔偿因管理人违法违规、违反基金合同、操作错误等给基金财产或者投资者造成的损失。"
    }
]

# 定义上下文QA模板
CONTEXT_QA_TMPL = """
你是私募基金问答助手。请根据以下信息回答问题：

信息：{context}
问题：{query}
"""
CONTEXT_QA_PROMPT = PromptTemplate(
    input_variables=["query", "context"],
    template=CONTEXT_QA_TMPL,
)

"""
Step4. 知识边界处理
专门设计了处理超出知识库范围问题的机制：
1. 问题主题识别：识别用户问题涉及的具体主题
2. 明确边界区分：在回答中明确区分知识库内容和模型知识
3. 提供有价值建议：引导用户寻求更权威的信息来源
"""
# 定义超出知识库范围问题的回答模板
OUTSIDE_KNOWLEDGE_TMPL = """
你是私募基金问答助手。用户的问题是关于私募基金的，但我们的知识库中没有直接相关的信息。
请首先明确告知用户"对不起，在我的知识库中没有关于[具体主题]的详细信息"，
然后，如果你有相关知识，可以以"根据我的经验"或"一般来说"等方式提供一些通用信息，
并建议用户查阅官方资料或咨询专业人士获取准确信息。

用户问题：{query}
缺失的知识主题：{missing_topic}
"""
OUTSIDE_KNOWLEDGE_PROMPT = PromptTemplate(
    input_variables=["query", "missing_topic"],
    template=OUTSIDE_KNOWLEDGE_TMPL,
)

"""
Step2. 工具设计,系统实现了三种核心工具：
1. 关键词搜索工具：通过关键词匹配检索相关规则，search_rules_by_keywords
2. 类别查询工具：根据规则类别查询相关规则，search_rules_by_category
3. 直接回答工具：基于问题与规则的匹配度直接回答用户问题，answer_question
"""
# 私募基金问答数据源
class FundRulesDataSource:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.rules_db = FUND_RULES_DB

    # 工具1：通过关键词搜索相关规则
    def search_rules_by_keywords(self, keywords: str) -> str:
        """通过关键词搜索相关私募基金规则"""
        keywords = keywords.strip().lower()
        keyword_list = re.split(r'[,，\s]+', keywords)
        
        matched_rules = []
        for rule in self.rules_db:
            rule_text = (rule["category"] + " " + rule["question"]).lower()
            match_count = sum(1 for kw in keyword_list if kw in rule_text)
            if match_count > 0:
                matched_rules.append((rule, match_count))
        
        matched_rules.sort(key=lambda x: x[1], reverse=True)
        
        if not matched_rules:
            return "未找到与关键词相关的规则。"
        
        result = []
        for rule, _ in matched_rules[:2]:
            result.append(f"类别: {rule['category']}\n问题: {rule['question']}\n答案: {rule['answer']}")
        
        return "\n\n".join(result)

    # 工具2：根据规则类别查询
    def search_rules_by_category(self, category: str) -> str:
        """根据规则类别查询私募基金规则"""
        category = category.strip()
        matched_rules = []
        
        for rule in self.rules_db:
            if category.lower() in rule["category"].lower():
                matched_rules.append(rule)
        
        if not matched_rules:
            return f"未找到类别为 '{category}' 的规则。"
        
        result = []
        for rule in matched_rules:
            result.append(f"问题: {rule['question']}\n答案: {rule['answer']}")
        
        return "\n\n".join(result)

    # 工具3：直接回答用户问题
    def answer_question(self, query: str) -> str:
        """直接回答用户关于私募基金的问题"""
        query = query.strip()
        
        best_rule = None
        best_score = 0
        
        for rule in self.rules_db:
            # 将用户输入的查询字符串转换为小写，然后按空白字符分割成单词列表，
            # 最后将该列表转换为集合，以去除重复单词并方便后续的集合操作。
            # 此操作是为了计算查询词与规则中的问题和类别之间的匹配度。
            query_words = set(query.lower().split())
            rule_words = set((rule["question"] + " " + rule["category"]).lower().split())
            # 使用集合的 intersection 方法计算 query_words 和 rule_words 的交集，
            # 即找出同时存在于查询词集合和规则词集合中的单词，
            # 结果存储在 common_words 变量中，用于后续计算匹配得分。
            common_words = query_words.intersection(rule_words)
            
            # 计算匹配得分，该得分表示查询词与规则中的问题和类别之间的匹配程度。
            # len(common_words) 表示查询词和规则词中共同出现的单词数量。
            # len(query_words) 表示查询词的总数量，使用 max(1, len(query_words)) 确保分母不为 0，避免除零错误。
            # 最终得分是共同单词数量在查询词总数中的占比，占比越高表示匹配度越高。
            score = len(common_words) / max(1, len(query_words))
            if score > best_score:
                best_score = score
                best_rule = rule
        
        # 知识边界处理:专门设计了处理超出知识库范围问题的机制
        if best_score < 0.2 or best_rule is None:
            # 识别问题主题
            missing_topic = self._identify_missing_topic(query)
            prompt = OUTSIDE_KNOWLEDGE_PROMPT.format(
                query=query,
                missing_topic=missing_topic
            )
            # 直接通过LLM获取回答可能导致输出格式与Agent期望不符
            # 将回答包装为AgentFinish格式而不是返回给Agent处理
            response = self.llm(prompt)
            # 返回格式化后的回答，让Agent直接返回最终结果
            return f"这个问题超出了知识库范围。\n\n{response}"
        
        context = best_rule["answer"]   
        # 格式化上下文，包含问题和答案
        context = f"问题: {best_rule['question']}\n答案: {context}"
        # 此代码行使用 CONTEXT_QA_PROMPT 模板来格式化提示信息。
        # CONTEXT_QA_PROMPT 是一个预定义的 PromptTemplate 对象，其模板字符串为 CONTEXT_QA_TMPL。
        # 通过调用 format 方法，将用户的查询语句 query 和相关上下文信息 context 填充到模板中，
        # 生成一个完整的提示文本，用于后续传递给大语言模型（LLM）以获取回答。
        prompt = CONTEXT_QA_PROMPT.format(query=query, context=context)
        
        return self.llm(prompt)
    
    def _identify_missing_topic(self, query: str) -> str:
        """识别查询中缺失的知识主题"""
        # 简单的主题提取逻辑
        query = query.lower()
        if "投资" in query and "资产" in query:
            return "私募基金可投资的资产类别"
        elif "公募" in query and "区别" in query:
            return "私募基金与公募基金的区别"
        elif "退出" in query and ("机制" in query or "方式" in query):
            return "创业投资基金的退出机制"
        elif "费用" in query and "结构" in query:
            return "私募基金的费用结构"
        elif "托管" in query:
            return "私募基金资产托管"
        # 如果无法确定具体主题，使用通用表述
        return "您所询问的具体主题"


# 定义Agent模板
AGENT_TMPL = """你是一个私募基金问答助手，请根据用户的问题选择合适的工具来回答。

你可以使用以下工具：

{tools}

按照以下格式回答问题：

---
Question: 用户的问题
Thought: 我需要思考如何回答这个问题
Action: 工具名称
Action Input: 工具的输入
Observation: 工具返回的结果
...（这个思考/行动/行动输入/观察可以重复几次）
Thought: 现在我知道答案了
Final Answer: 给用户的最终答案
---

注意：
1. 如果知识库中没有相关信息，请明确告知用户"对不起，在我的知识库中没有关于[具体主题]的详细信息"
2. 如果你基于自己的知识提供补充信息，请用"根据我的经验"或"一般来说"等前缀明确标识
3. 回答要专业、简洁、准确

Question: {input}
{agent_scratchpad}
"""


"""
Step3. Agent架构搭建
基于LangChain的Agent框架，构建了完整的反应式智能体系统：
1. 自定义提示模板(CustomPromptTemplate)：定义Agent的思考和决策格式
2. 自定义输出解析器(CustomOutputParser)：解析LLM的输出，确定下一步行动
3. Agent执行器(AgentExecutor)：协调智能体与工具的交互
"""
# 自定义Prompt模板
class CustomPromptTemplate(StringPromptTemplate):
    """
    自定义提示模板类，继承自 StringPromptTemplate。
    该类用于生成符合特定格式的提示信息，支持在模板中动态插入工具信息和中间推理步骤。
    """
    template: str  # 提示模板字符串，用于定义提示信息的基本结构
    tools: List[Tool]  # 可用工具列表，每个工具包含名称和描述

    def format(self, **kwargs) -> str:
        """
        根据输入的参数和中间推理步骤，格式化生成最终的提示信息。

        参数:
            **kwargs: 包含输入参数和中间推理步骤的关键字参数。
                      其中 "intermediate_steps" 是必需的，代表智能体的中间推理步骤。

        返回:
            str: 格式化后的完整提示信息字符串。
        """
        # 从 kwargs 中移除中间推理步骤，并存储在 intermediate_steps 变量中
        intermediate_steps = kwargs.pop("intermediate_steps")
        # 初始化 thoughts 字符串，用于存储智能体的思考过程和观察结果
        thoughts = ""
        # 遍历中间推理步骤，每个步骤包含一个动作和对应的观察结果
        for action, observation in intermediate_steps:
            # 将动作的日志信息添加到 thoughts 中
            thoughts += action.log
            # 将观察结果添加到 thoughts 中，并在其后添加新的思考提示
            thoughts += f"\nObservation: {observation}\nThought: "
        
        # 将整理好的思考过程存储到 kwargs 中，键名为 "agent_scratchpad"
        kwargs["agent_scratchpad"] = thoughts
        # 将工具列表中的每个工具名称和描述拼接成字符串，用换行符分隔，存储到 kwargs 中
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        
        # 使用整理好的 kwargs 对模板字符串进行格式化，生成最终的提示信息
        return self.template.format(**kwargs)


# 自定义输出解析器
class CustomOutputParser(AgentOutputParser):
    """
    自定义输出解析器，继承自 AgentOutputParser 类。
    该类用于解析大语言模型（LLM）的输出，根据输出内容判断返回 AgentAction 或 AgentFinish 对象。
    """
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        """
        解析大语言模型的输出，根据输出内容返回相应的动作或最终答案。

        参数:
            llm_output (str): 大语言模型生成的输出字符串。

        返回:
            Union[AgentAction, AgentFinish]: 根据解析结果返回 AgentAction 或 AgentFinish 对象。
                                            AgentAction 表示需要执行的动作，
                                            AgentFinish 表示已经得到最终答案。

        异常:
            ValueError: 当无法解析 LLM 输出且输出内容较短时抛出。
        """
        # 打印LLM输出，当前代码行被注释，实际使用时可取消注释用于调试
        #print(f"LLM输出: {llm_output}")
        
        # 场景1：输出包含"Final Answer:"，表明已经得到最终答案
        # 直接处理为最终答案，提取"Final Answer:"之后的内容作为返回值
        if "Final Answer:" in llm_output:
            # 使用 split 方法分割字符串，取最后一部分并去除首尾空白
            final_answer = llm_output.split("Final Answer:")[-1].strip()
            return AgentFinish(
                return_values={"output": final_answer},  # 返回最终答案
                log=llm_output,  # 记录完整的LLM输出
            )
            
        # 场景2：输出以"对不起"或"抱歉"开头，可能是LLM直接给出了回答而非遵循格式
        # 直接将其视为最终答案
        if llm_output.strip().startswith("对不起") or llm_output.strip().startswith("抱歉"):
            return AgentFinish(
                return_values={"output": llm_output.strip()},  # 返回原输出内容
                log=f"Direct response detected: {llm_output}"  # 记录检测到直接响应
            )
            
        # 场景3：输出包含明确的知识边界声明，也视为最终答案
        # 定义知识边界声明短语列表
        knowledge_boundary_phrases = [
            "在我的知识库中没有",
            "超出了我的知识范围",
            "我没有相关信息",
            "根据我的经验"
        ]
        
        # 遍历知识边界声明短语，检查输出中是否包含这些短语
        for phrase in knowledge_boundary_phrases:
            if phrase in llm_output:
                return AgentFinish(
                    return_values={"output": llm_output.strip()},  # 返回原输出内容
                    log=f"Knowledge boundary response detected: {llm_output}"  # 记录检测到知识边界响应
                )

        # 场景4：尝试解析Action和Action Input，表明需要执行某个工具动作
        # 定义正则表达式模式，用于匹配 Action 和 Action Input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        # 使用正则表达式搜索LLM输出
        match = re.search(regex, llm_output, re.DOTALL)
        
        # 如果没有匹配到 Action 和 Action Input
        if not match:
            # 如果输出内容较长（超过50个字符），可能是LLM直接给出了详细回答
            if len(llm_output.strip()) > 50:  # 假设超过50个字符的回答为实质性内容
                return AgentFinish(
                    return_values={"output": llm_output.strip()},  # 返回原输出内容
                    log=f"Long unstructured response detected: {llm_output}"  # 记录检测到长无结构响应
                )
            # 真正无法解析的情况，抛出异常
            raise ValueError(f"无法解析LLM输出: `{llm_output}`")
        
        # 从匹配结果中提取 Action 和 Action Input
        action = match.group(1).strip()  # 提取工具名称
        action_input = match.group(2)  # 提取工具输入
        
        # 返回 AgentAction 对象，表示需要执行指定工具和输入
        return AgentAction(
            tool=action,  # 指定工具名称
            tool_input=action_input.strip(" ").strip('"'),  # 去除输入前后的空白和引号
            log=llm_output  # 记录完整的LLM输出
        )


def create_fund_qa_agent():
    # 定义LLM
    llm = Tongyi(model_name="Qwen-Turbo-2025-04-28", dashscope_api_key=DASHSCOPE_API_KEY)
    
    # 创建数据源
    fund_rules_source = FundRulesDataSource(llm)
    
    # 定义工具
    tools = [
        Tool(
            name="关键词搜索",
            func=fund_rules_source.search_rules_by_keywords,
            description="当需要通过关键词搜索私募基金规则时使用，输入应为相关关键词",
        ),
        Tool(
            name="类别查询",
            func=fund_rules_source.search_rules_by_category,
            description="当需要查询特定类别的私募基金规则时使用，输入应为类别名称。类别名称有两种：设立与募集, 监管规定",
        ),
        Tool(
            name="回答问题",
            func=fund_rules_source.answer_question,
            description="当能够直接回答用户问题时使用，输入应为完整的用户问题",
        ),
    ]
    
    # 创建Agent提示模板
    agent_prompt = CustomPromptTemplate(
        template=AGENT_TMPL,
        tools=tools,
        input_variables=["input", "intermediate_steps"],
    )
    
    # 创建输出解析器
    output_parser = CustomOutputParser()
    
    # 创建LLM链
    llm_chain = LLMChain(llm=llm, prompt=agent_prompt)
    
    # 获取工具名称
    tool_names = [tool.name for tool in tools]
    
    # 创建Agent
    # 此段代码创建了一个 LLMSingleActionAgent 实例，该实例是 LangChain 中用于单步动作决策的智能体。
    # LLMSingleActionAgent 能够根据输入的提示信息，借助 LLM 进行思考和决策，每次仅选择一个工具来执行。
    # 参数说明：
    # - llm_chain: 传入之前创建的 LLMChain 实例，该实例包含了语言模型和提示模板，用于生成思考过程和决策。
    # - output_parser: 传入自定义的输出解析器，用于解析 LLM 的输出，判断是执行工具动作还是返回最终答案。
    # - stop: 指定停止生成文本的标记，当 LLM 生成的内容遇到 "\nObservation:" 时停止。
    # - allowed_tools: 传入允许使用的工具名称列表，限制智能体只能选择列表中的工具。
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names,
    )
    
    # 创建Agent执行器
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True,handle_parsing_errors=True,
            max_iterations=5
    )
    
    return agent_executor


if __name__ == "__main__":
    # 创建Agent
    fund_qa_agent = create_fund_qa_agent()
    
    print("=== 私募基金运作指引问答助手（反应式智能体）===\n")
    print("使用模型：Qwen-Turbo-2025-04-28\n")
    print("您可以提问关于私募基金的各类问题，输入'退出'结束对话\n")
    
    # 主循环
    while True:
        try:
            user_input = input("请输入您的问题：")
            if user_input.lower() in ['退出', 'exit', 'quit']:
                print("感谢使用，再见！")
                break
            
            response = fund_qa_agent.run(user_input)
            print(f"回答: {response}\n")
            print("-" * 40)
        except KeyboardInterrupt:
            print("\n程序已中断，感谢使用！")
            break
        except Exception as e:
            print(f"发生错误：{e}")
            print("请尝试重新提问或更换提问方式。") 