"""
多文档并行问答，使用ParallelDocQA
ParallelDocQA 的适用场景：
    • 支持多文档、多格式（PDF/Word/PPT/TXT/HTML）并行检索与问答。
    • 适合大体量文档、复杂结构材料的高效问答。
    • 需要并行处理、RAG召回、自动摘要等场景。

ParallelDocQA的并行处理机制：
    • 采用 parallel_exec 对每个文档 chunk 并行分发 QA 任务，大幅提升处理效率。
    • 每个 chunk 独立调用 ParallelDocQAMember 进行问答，互不影响。
    • 支持多轮重试，自动过滤无效响应，保证结果质量。

ParallelDocQA 具体执行流程：
    Step1，分片并行问答，每个 chunk 由 _ask_member_agent 调用 LLM 进行独立问答，得到 68 个回答结果。
    Step2，结果收集与过滤，收集所有分片的回答，过滤掉无效或空响应，只保留有用的答案片段。
    Step3，结果汇总与检索，将所有有效分片的回答拼接、汇总，作为“member_res”。
        后续还会用这些回答生成检索关键词，再对原始文档做一次更精准的检索（RAG召回）。
    Step4，最终摘要，最后将检索到的内容交给摘要 Agent，生成最终的综合答案。

注：每个分片一次 LLM 问答，全部并行，总数等于分片数。
   所有分片的结果会被汇总、过滤、再进一步检索和摘要，最终输出给用户。
"""
# https://arxiv.org/pdf/2310.08560.pdf 这里适用文档 QWEN_TECHNICAL_REPORT.pdf

from qwen_agent.agents.doc_qa import ParallelDocQA
from qwen_agent.gui import WebUI
import os
import dashscope
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', 'sk-58f051ae745e4bb19fdca31735105b11')
def test():
    bot = ParallelDocQA(llm={'model': 'qwen2.5-72b-instruct', 'generate_cfg': {'max_retries': 10}})
    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'text': '介绍实验方法'
                },
                {
                    'file': 'https://arxiv.org/pdf/2310.08560.pdf'
                },
            ]
        },
    ]
    for rsp in bot.run(messages):
        print('bot response:', rsp)


def app_gui():
    # Define the agent
    bot = ParallelDocQA(
        llm={
            'model': 'qwen-turbo-latest', #qwen2.5-72b-instruct
            #'dashscope_api_key': '你的API_KEY',
            'generate_cfg': {
                'max_retries': 10
            }
        },
        description='并行QA后用RAG召回内容并回答。支持文件类型：PDF/Word/PPT/TXT/HTML。使用与材料相同的语言提问会更好。',
    )

    chatbot_config = {'prompt.suggestions': [{'text': '介绍实验方法'}]}

    WebUI(bot, chatbot_config=chatbot_config).run()


if __name__ == '__main__':
    # test()
    app_gui()
