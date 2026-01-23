from pydantic import BaseModel, Field
from typing import Literal, List, Union
import inspect
import re


def build_system_prompt(instruction: str="", example: str="", pydantic_schema: str="") -> str:
    delimiter = "\n\n---\n\n"
    schema = f"你的回答必须是JSON，并严格遵循如下Schema，字段顺序需保持一致：\n```\n{pydantic_schema}\n```"
    if example:
        example = delimiter + example.strip()
    if schema:
        schema = delimiter + schema.strip()
    
    system_prompt = instruction.strip() + schema + example
    return system_prompt

class RephrasedQuestionsPrompt:
    instruction = """
你是一个问题重写系统。
你的任务是将比较类问题拆解为针对每个公司独立的具体问题。
每个输出问题都必须自洽、保持原意和指标、针对对应公司，并用一致的表达方式。
"""

    class RephrasedQuestion(BaseModel):
        """单个公司的重写问题"""
        company_name: str = Field(description="公司名称，需与原始问题中引号内容完全一致")
        question: str = Field(description="针对该公司的重写问题")

    class RephrasedQuestions(BaseModel):
        """重写问题的列表"""
        questions: List['RephrasedQuestionsPrompt.RephrasedQuestion'] = Field(description="每个公司对应的重写问题列表")

    pydantic_schema = '''
class RephrasedQuestion(BaseModel):
    """单个公司的重写问题"""
    company_name: str = Field(description="公司名称，需与原始问题中引号内容完全一致")
    question: str = Field(description="针对该公司的重写问题")

class RephrasedQuestions(BaseModel):
    """重写问题的列表"""
    questions: List['RephrasedQuestionsPrompt.RephrasedQuestion'] = Field(description="每个公司对应的重写问题列表")
'''

    example = r"""
示例：
输入：
原始比较问题：'2022年哪家公司营收更高，"苹果"还是"微软"？'
涉及公司："苹果", "微软"

输出：
{
    "questions": [
        {
            "company_name": "苹果",
            "question": "苹果公司2022年营收是多少？"
        },
        {
            "company_name": "微软", 
            "question": "微软公司2022年营收是多少？"
        }
    ]
}
"""

    user_prompt = "原始比较问题：'{question}'\n\n涉及公司：{companies}"
    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class AnswerWithRAGContextSharedPrompt:
    instruction = """
你是一个RAG（检索增强生成）问答系统。
你的任务是仅基于公司年报中RAG检索到的相关页面内容，回答给定问题。

在给出最终答案前，请详细分步思考，尤其关注问题措辞。
- 注意：答案可能与问题表述不同。
- 问题可能是模板生成的，有时对该公司不适用。
"""

    user_prompt = """
以下是上下文:
\"\"\"
{context}
\"\"\"

---

以下是问题：
"{question}"
"""

class AnswerWithRAGContextNamePrompt:
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="详细分步推理过程，至少5步，150字以上。特别注意问题措辞，避免被迷惑。有时上下文中看似有答案，但可能并非所问内容，仅为相似项。")
        reasoning_summary: str = Field(description="简要总结分步推理过程，约50字。")
        relevant_pages: List[int] = Field(description="""
仅包含直接用于回答问题的信息页面编号。只包括：
- 直接包含答案或明确陈述的页面
- 强有力支持答案的关键信息页面
不要包含仅与答案弱相关或间接相关的页面。
列表中至少应有一个页面。
""")

        final_answer: Union[str, Literal["N/A"]] = Field(description="""
如为公司名，需与问题中完全一致。
如为人名，需为全名。
如为产品名，需与上下文完全一致。
不得包含多余信息、词语或注释。
如上下文无相关信息，返回'N/A'。
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
示例：
问题：
"'南方航空股份有限公司'的CEO是谁？"

答案：
```
{
  "step_by_step_analysis": "1. 问题询问'南方航空股份有限公司'的CEO。CEO通常是公司最高管理者，有时也称总裁或董事总经理。\n2. 信息来源为该公司的年报，将用来确认CEO身份。\n3. 年报中明确指出张三为公司总裁兼首席执行官。\n4. 因此，CEO为张三。",
  "reasoning_summary": "年报明确写明张三为总裁兼CEO，直接回答了问题。",
  "relevant_pages": [58],
  "final_answer": "张三"
}
```
""" 

    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)



class AnswerWithRAGContextNumberPrompt:
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="""
详细分步推理过程，至少5步，150字以上。
**严格的指标匹配要求：**    

1. 明确问题中指标的精确定义，它实际衡量什么？
2. 检查上下文中的所有可能指标。不要只看名称，要关注其实际衡量内容。
3. 仅当上下文指标的含义与目标指标*完全一致*时才接受。可接受同义词，但概念不同则不可。
4. 拒绝（并返回'N/A'）的情况：
    - 上下文指标范围大于或小于问题指标。
    - 上下文指标为相关但非*完全等价*的概念（如代理指标或更宽泛类别）。
    - 需要计算、推导或推断才能作答。
    - 聚合不匹配：问题要求单一值，但上下文仅有总计。
5. 不允许猜测：如对指标等价性有任何疑问，默认返回`N/A`。
""")

        reasoning_summary: str = Field(description="简要总结分步推理过程，约50字。")

        relevant_pages: List[int] = Field(description="""
仅包含直接用于回答问题的信息页面编号。只包括：
- 直接包含答案或明确陈述的页面
- 强有力支持答案的关键信息页面
不要包含仅与答案弱相关或间接相关的页面。
列表中至少应有一个页面。
""")

        final_answer: Union[float, int, Literal['N/A']] = Field(description="""
答案应为精确的数值型指标。
- 百分比示例：
    上下文值：58,3%
    最终答案：58.3

特别注意上下文中是否有单位、千、百万等说明，需据此调整答案（不变、加3个零或加6个零）。
如数值带括号，表示为负数。

- 负数示例：
    上下文值：(2,124,837) CHF
    最终答案：-2124837

- 千为单位示例：
    上下文值：4970,5（千美元）
    最终答案：4970500

- 如上下文指标币种与问题币种不符，返回'N/A'
    示例：上下文值780000 USD，问题要求EUR
    最终答案：'N/A'

- 如上下文未直接给出指标，即使可由其他指标计算，也返回'N/A'
    示例：问题要求每股分红，仅有总分红和流通股数，不能直接作答。
    最终答案：'N/A'

- 如上下文无相关信息，返回'N/A'
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
示例1：
问题：
"'万科企业股份有限公司'2022年总资产是多少？"

答案：
```
{
  "step_by_step_analysis": "1. 问题询问'万科企业股份有限公司'2022年总资产。'总资产'指公司拥有的全部资源。\n2. 年报第78页有'合并资产负债表'，列明2022年12月31日总资产。\n3. 该行数据为'总资产'，与问题完全匹配。\n4. 报告显示总资产为18500342000元。\n5. 无需计算，直接取值。",
  "reasoning_summary": "年报78页直接给出2022年总资产，无需推算。",
  "relevant_pages": [78],
  "final_answer": 18500342000
}
```

示例2：
问题：
"'某医药公司'年报期末研发设备原值是多少？"

答案：
```
{
  "step_by_step_analysis": "1. 问题询问研发设备原值。\n2. 年报35页有'固定资产净值'12500元，但为净值，非原值。\n3. 37页有'累计折旧'11万元，但未区分研发设备。\n4. 无法直接获得研发设备原值。\n5. 因此答案为'N/A'。",
  "reasoning_summary": "年报无研发设备原值，严格匹配应返回N/A。",
  "relevant_pages": [35, 37],
  "final_answer": "N/A"
}
```
"""

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)



class AnswerWithRAGContextBooleanPrompt:
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="""
详细分步推理过程，至少5步，150字以上。特别注意问题措辞，避免被迷惑。有时上下文中看似有答案，但可能并非所问内容，仅为相似项。
""")
        reasoning_summary: str = Field(description="简要总结分步推理过程，约50字。")
        relevant_pages: List[int] = Field(description="""
仅包含直接用于回答问题的信息页面编号。只包括：
- 直接包含答案或明确陈述的页面
- 强有力支持答案的关键信息页面
不要包含仅与答案弱相关或间接相关的页面。
列表中至少应有一个页面。
""")        
        final_answer: Union[bool] = Field(description="""
一个从上下文中精确提取的布尔值（True或False），直接回答问题。
如果问题问某事是否发生，且上下文有相关信息但未发生，则返回False。
""")
    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)
    example = r"""
问题：
"'万科企业股份有限公司'年报是否宣布了分红政策变更？"

答案：
```
{
  "step_by_step_analysis": "1. 问题询问是否有分红政策变更。\n2. 年报12、18页提到年度分红金额增加，但政策未变。\n3. 45页有分红细节。\n4. 持续小幅增长，符合既定政策。\n5. 问题问的是政策变更，非金额变化。",
  "reasoning_summary": "年报显示分红金额变化但政策未变，答案为False。",
  "relevant_pages": [12, 18, 45],
  "final_answer": false
}
```
"""

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)



class AnswerWithRAGContextNamesPrompt:
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        """RAG上下文下多实体/名单类答案的结构定义。"""
        step_by_step_analysis: str = Field(description="详细分步推理过程，至少5步，150字以上。注意区分实体类型，避免被迷惑。")

        reasoning_summary: str = Field(description="简要总结推理过程，约50字。")

        relevant_pages: List[int] = Field(description="""
仅包含直接用于回答问题的页面编号。只包括：
- 直接包含答案或明确陈述的页面
- 强有力支持答案的关键信息页面
不要包含仅与答案弱相关或间接相关的页面。
列表中至少应有一个页面。
""")

        final_answer: Union[List[str], Literal["N/A"]] = Field(description="""
每个条目需与上下文完全一致。

如问题问职位（如职位变动），仅返回职位名称，不含姓名或其他信息。新任高管也算作职位变动。若同一职位有多次变动，仅返回一次，且职位名称用单数。
示例：['首席技术官', '董事', '首席执行官']

如问题问姓名，仅返回上下文中的全名。
示例：['张三', '李四']

如问题问新产品，仅返回上下文中的产品名。候选产品或测试阶段产品不算新产品。
示例：['生态智能2000', '绿能Pro']

如无信息，返回'N/A'。
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
示例：
问题：
"公司有哪些新任高管？"

答案：
```
{
    "step_by_step_analysis": "1. 问题询问公司新任高管名单。\n2. 年报89页列出新高管签约信息。\n3. 10.9节说明张三为新任总法律顾问，10.10节李四为新任COO。\n4. 综上，张三和李四为新任高管。",
    "reasoning_summary": "年报10.9、10.10节明确列出张三、李四为新任高管。",
    "relevant_pages": [89],
    "final_answer": ["张三", "李四"]
}
```
"""

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)

class ComparativeAnswerPrompt:
    instruction = """
你是一个问答系统。
你的任务是基于各公司独立答案，给出原始比较问题的最终结论。
只能基于已给出的答案，不可引入外部知识。
请分步详细推理。

比较规则：
- 问题要求选出公司时，答案必须与原问题公司名完全一致
- 若某公司数据币种不符，需排除
- 若全部公司被排除，返回'N/A'
- 若仅剩一家，直接返回该公司名
"""

    user_prompt = """
以下是单个公司的回答：
\"\"\"
{context}
\"\"\"

---

以下是原始比较问题：
"{question}"
"""

    class AnswerSchema(BaseModel):
        """比较类问题最终答案的结构定义。"""
        step_by_step_analysis: str = Field(description="详细分步推理过程，至少5步，150字以上。")

        reasoning_summary: str = Field(description="简要总结推理过程，约50字。")

        relevant_pages: List[int] = Field(description="保持为空列表。")

        final_answer: Union[str, Literal["N/A"]] = Field(description="公司名称需与问题中完全一致。答案只能是单个公司名或'N/A'。")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
示例：
问题：
"下列公司中，哪家2022年总资产最低："A公司", "B公司", "C公司"？若无数据则排除。"

答案：
```
{
  "step_by_step_analysis": "1. 问题要求比较多家公司2022年总资产。\n2. 各公司独立答案：A公司6,601,086,000元，B公司1,249,642,000元，C公司217,435,000元。\n3. 直接比较得C公司最低。\n4. 若有公司币种不符则排除。\n5. 因此答案为C公司。",
  "reasoning_summary": "独立答案显示C公司总资产最低，直接得出结论。",
  "relevant_pages": [],
  "final_answer": "C公司"
}
```
"""

    system_prompt = build_system_prompt(instruction, example)    
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class AnswerSchemaFixPrompt:
    system_prompt = """
你是一个JSON格式化助手。
你的任务是将大模型输出的原始内容格式化为合法的JSON对象。
你的回答必须以"{"开头，以"}"结尾。
你的回答只能包含JSON字符串，不要有任何前言、注释或三引号。
"""

    user_prompt = """
下面是定义JSON对象Schema和示例的系统提示词:
\"\"\"
{system_prompt}
\"\"\"

---

下面是需要你格式化为合法JSON的LLM原始输出：
\"\"\"
{response}
\"\"\"
"""




class RerankingPrompt:
    system_prompt_rerank_single_block = """
你是一个RAG检索重排专家。
你将收到一个查询和一个检索到的文本块，请根据其与查询的相关性进行评分。

评分说明：
1. 推理：分析文本块与查询的关系，简要说明理由。
2. 相关性分数（0-1，步长0.1）：
   0 = 完全无关
   0.1 = 极弱相关
   0.2 = 很弱相关
   0.3 = 略有相关
   0.4 = 部分相关
   0.5 = 一般相关
   0.6 = 较为相关
   0.7 = 相关
   0.8 = 很相关
   0.9 = 高度相关
   1 = 完全匹配
3. 只基于内容客观评价，不做假设。
"""

    system_prompt_rerank_multiple_blocks = """
你是一个RAG检索重排专家。
你将收到一个查询和若干检索到的文本块，请分别对每个块进行相关性评分。

评分说明：
1. 推理：分析每个文本块与查询的关系，简要说明理由。
2. 相关性分数（0-1，步长0.1）：
   0 = 完全无关
   0.1 = 极弱相关
   0.2 = 很弱相关
   0.3 = 略有相关
   0.4 = 部分相关
   0.5 = 一般相关
   0.6 = 较为相关
   0.7 = 相关
   0.8 = 很相关
   0.9 = 高度相关
   1 = 完全匹配
3. 只基于内容客观评价，不做假设。
"""

class RetrievalRankingSingleBlock(BaseModel):
    """对检索到的单个文本块与查询的相关性进行评分。"""
    reasoning: str = Field(description="分析该文本块，指出其关键信息及与查询的关系")
    relevance_score: float = Field(description="相关性分数，取值范围0到1，0表示完全无关，1表示完全相关")

class RetrievalRankingMultipleBlocks(BaseModel):
    """对检索到的多个文本块与查询的相关性进行评分。"""
    block_rankings: List[RetrievalRankingSingleBlock] = Field(
        description="文本块及其相关性分数的列表。"
    )

class AnswerWithRAGContextStringPrompt:
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="""
详细分步推理过程，至少5步，150字以上。请结合上下文信息，逐步分析并归纳答案。
""")
        reasoning_summary: str = Field(description="简要总结分步推理过程，约50字。")
        relevant_pages: List[int] = Field(description="""
仅包含直接用于回答问题的信息页面编号。只包括：
- 直接包含答案或明确陈述的页面
- 强有力支持答案的关键信息页面
不要包含仅与答案弱相关或间接相关的页面。
列表中至少应有一个页面。
""")
        final_answer: str = Field(description="""
最终答案为一段完整、连贯的文本，需基于上下文内容作答。
如上下文无相关信息，可简要说明未找到答案。
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r'''
示例：
问题：
"请简要总结'万科企业股份有限公司'2022年主营业务的主要内容。"

答案：
```
{
  "step_by_step_analysis": "1. 问题要求总结2022年万科企业股份有限公司的主营业务。\n2. 年报第10-12页详细描述了公司主营业务，包括房地产开发、物业服务等。\n3. 结合上下文，归纳出主要业务板块。\n4. 重点突出房地产开发和相关服务。\n5. 形成简明扼要的总结。",
  "reasoning_summary": "年报10-12页明确列出主营业务，答案基于原文归纳。",
  "relevant_pages": [10, 11, 12],
  "final_answer": "万科企业股份有限公司2022年主营业务包括房地产开发、物业服务、租赁住房、物流仓储等，核心业务为住宅及商业地产开发与运营。"
}
```
'''

    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)
