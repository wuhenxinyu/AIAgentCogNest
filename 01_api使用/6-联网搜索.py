from email import message
import os
from openai import OpenAI

os.environ['DASHSCOPE_API_KEY'] = "sk-58f051ae745e4bb19fdca31735105b11"
# 从环境变量中，获取 DASHSCOPE_API_KEY
api_key = os.environ.get('DASHSCOPE_API_KEY')

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=api_key, 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务的base_url
)
# 创建一个聊天完成请求
completion = client.chat.completions.create(
    # 指定使用的模型为qwen-plus
    model="qwen-plus",  # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    
    # 设置对话消息列表
    messages=[
        # 系统消息，设置AI助手的角色
        {'role': 'system', 'content': '你是一个乐于助人的智能助手。'},
        # 用户消息，询问关于中国队在巴黎奥运会的金牌数量
        # {'role': 'user', 'content': '中国队在巴黎奥运会获得了多少枚金牌，巴黎奥运会什么时候举行的'}
        {'role': 'user', 'content': '怎么写好一个技术方案？'}
    ],
    
    # 额外参数配置
    extra_body={
        "enable_search": True  # 启用联网搜索功能，使模型能够获取实时信息
    }
)
# 将API返回的完整响应转换为JSON格式并打印输出
# indent=2 表示JSON输出时使用2个空格的缩进，使输出更易读
print(completion.model_dump_json(indent=2))

# 打印模型返回的消息内容
print('message:', completion.choices[0].message.content)
