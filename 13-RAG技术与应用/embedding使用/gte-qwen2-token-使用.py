
# 导入必要的库
import torch  # PyTorch深度学习库
import torch.nn.functional as F  # PyTorch函数式接口，包含各种神经网络函数

from torch import Tensor  # 导入Tensor类型，用于类型提示
from modelscope import AutoTokenizer, AutoModel  # 从modelscope导入自动分词器和模型加载器


# 定义最后一个token池化函数
# 该函数从最后的隐藏状态中提取每个序列的最后一个有效token的表示
def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    # 检查是否为左侧填充（即所有序列最后一个位置都有效）
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        # 如果是左侧填充，直接返回最后一个位置的隐藏状态
        return last_hidden_states[:, -1]
    else:
        # 如果是右侧填充，计算每个序列的实际长度（减1是因为索引从0开始）
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        # 返回每个序列最后一个有效token的隐藏状态
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


# 定义获取详细指令的函数
# 将任务描述和查询组合成特定格式的指令
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


# 每个查询都必须附带一个描述任务的简短指令
# 定义任务描述：给定网络搜索查询，检索相关的回答段落
task = 'Given a web search query, retrieve relevant passages that answer the query'
# 创建查询列表，每个查询都通过get_detailed_instruct函数添加了任务描述
queries = [
    get_detailed_instruct(task, 'how much protein should a female eat'),  # 女性应该摄入多少蛋白质
    get_detailed_instruct(task, 'summit define')  # summit（顶峰）的定义
]
# 检索文档不需要添加指令
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",  # 关于女性蛋白质摄入量的文档
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."  # 关于summit定义的文档
]
# 将查询和文档合并为一个输入文本列表
input_texts = queries + documents

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 如果有GPU，显示GPU信息
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")

# 设置模型路径
model_dir = "/root/autodl-tmp/models/iic/gte_Qwen2-1.5B-instruct"
# 加载分词器，trust_remote_code=True允许使用远程代码
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
# 加载模型，并指定设备为GPU
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(device)

# 设置最大序列长度
max_length = 8192

# 对输入文本进行分词处理
# padding=True：对较短的序列进行填充，使批次中所有序列长度一致
# truncation=True：截断超过max_length的序列
# return_tensors='pt'：返回PyTorch张量
batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
# 将输入数据移动到指定设备(GPU或CPU)
batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

# 将分词后的输入传入模型，获取输出
outputs = model(**batch_dict)
# 使用last_token_pool函数从最后的隐藏状态中提取每个序列的表示
embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# 对嵌入向量进行L2归一化，使其长度为1
# p=2表示L2范数，dim=1表示在第1维（特征维度）上进行归一化（归一化是将向量缩放到单位长度(模长为1)的过程。这里使用L2范数归一化，即将向量除以其L2范数(各分量平方和的平方根)。dim=1表示在特征维度上进行归一化操作。）
embeddings = F.normalize(embeddings, p=2, dim=1)
# 计算查询和文档之间的相似度分数
# embeddings[:2]：查询的嵌入向量（前两个）
# embeddings[2:]：文档的嵌入向量（后两个）
# .T：转置操作
# * 100：将相似度分数缩放到0-100的范围
scores = (embeddings[:2] @ embeddings[2:].T) * 100
# 打印相似度分数
print(scores.tolist())
# 结果解释:
# [[70.00666809082031, 8.184867858886719], [14.62420654296875, 77.71405792236328]]
# 70.00: 第一个查询(蛋白质摄入)与第一个文档(蛋白质指南)的相似度（高相关）
# 8.18: 第一个查询(蛋白质摄入)与第二个文档(summit定义)的相似度（低相关）
# 14.62: 第二个查询(summit定义)与第一个文档(蛋白质指南)的相似度（低相关）
# 77.71: 第二个查询(summit定义)与第二个文档(summit定义)的相似度（高相关）
