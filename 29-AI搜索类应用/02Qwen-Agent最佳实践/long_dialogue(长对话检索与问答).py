"""
长对话检索与问答，使用DialogueRetrievalAgent
DialogueRetrievalAgent 的适用场景：
• 超长对话、超长文本的问答场景。
• 用户输入内容极长，普通 LLM 无法直接处理时。
• 需要将长对话内容分片存储、检索，再进行问答。
"""
from qwen_agent.agents import DialogueRetrievalAgent
from qwen_agent.gui import WebUI
import dashscope
import os

dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', 'sk-58f051ae745e4bb19fdca31735105b11')

def test():
    # Define the agent
    # 初始化一个对话检索代理实例，指定使用的大语言模型为 'qwen-turbo-latest'
    # DialogueRetrievalAgent 是一个具备对话检索功能的代理，可用于处理包含大量文本的对话场景
    # llm 参数是一个字典，用于指定所使用的大语言模型
    bot = DialogueRetrievalAgent(llm={'model': 'qwen-turbo-latest'})

    # 准备聊天内容，构建一个长文本，模拟实际应用中包含大量干扰信息的场景
    # 使用 '，' 作为分隔符，将 1000 次重复的 '这是干扰内容'、关键信息 '小明的爸爸叫大头' 
    # 以及另外 1000 次重复的 '这是干扰内容' 拼接成一个长字符串
    long_text = '，'.join(['这是干扰内容'] * 1000 + ['小明的爸爸叫大头'] + ['这是干扰内容'] * 1000)
    
    # 构建用户消息列表，符合对话消息的格式要求
    # 消息是一个字典列表，每个字典包含 'role' 和 'content' 两个键
    # 'role' 表示消息发送者的角色，这里是 'user' 表示用户
    # 'content' 是消息的具体内容，包含用户的问题和前面构建的长文本
    messages = [{'role': 'user', 'content': f'小明爸爸叫什么？\n{long_text}'}]

    # 调用代理的 run 方法来处理用户消息并获取响应
    # run 方法返回一个可迭代对象，通过 for 循环逐次获取响应内容
    # 每次获取到响应后，将其打印输出，展示代理的回答
    for response in bot.run(messages):
        print('bot response:', response)


def app_tui():
    bot = DialogueRetrievalAgent(llm={'model': 'qwen-max'})

    # Chat
    messages = []
    while True:
        query = input('user question: ')
        messages.append({'role': 'user', 'content': query})
        response = []
        for response in bot.run(messages=messages):
            print('bot response:', response)
        messages.extend(response)


def app_gui():
    # Define the agent
    bot = DialogueRetrievalAgent(llm={'model': 'qwen-max'})

    WebUI(bot).run()


if __name__ == '__main__':
    test()
    # app_tui()
    # app_gui()
