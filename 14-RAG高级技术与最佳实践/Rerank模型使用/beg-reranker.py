"""
  在BGE-Rerank模型中，相关性分数scores是一个未归一化的对数几率（logits）值，范围没有固定的上限或下限（不像某些模型限制在0-1）。不过BGE-Rerank的分数通常落在以下范围：
        高相关性： 3.0~10.0
        中等相关性：0.0~3.0
        低相关性/不相关：负数（如-5.0以下）
"""
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('BAAI/bge-reranker-large', cache_dir='dev/shm/models')

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化分词器(tokenizer)，用于将文本转换为模型可以理解的数字序列
# 从预训练的BAAI/bge-reranker-large模型加载分词器
tokenizer = AutoTokenizer.from_pretrained('/dev/shm/models/BAAI/bge-reranker-large')

# 加载预训练的重排序模型
# AutoModelForSequenceClassification用于文本对相关性打分任务
# to(device)将模型移动到GPU(如果可用)或CPU上进行计算
model = AutoModelForSequenceClassification.from_pretrained('/dev/shm/models/BAAI/bge-reranker-large').to(device)
model.eval()

# 创建一个包含查询和文档的文本对列表
# pairs中每个元素是一个列表,包含[查询文本, 候选文档文本]
pairs = [['what is panda?', 'The giant panda is a bear species endemic to China.']]

# 使用tokenizer对文本对进行编码
# padding=True 确保所有序列长度一致
# truncation=True 对过长的序列进行截断
# return_tensors='pt' 返回PyTorch张量格式
inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt')

# 将输入数据移动到GPU(如果可用)
inputs = {k: v.to(device) for k, v in inputs.items()}

# 将编码后的输入传入模型进行相关性打分
# model(**inputs)将所有tokenizer编码得到的输入传给模型
# .logits获取模型输出的原始分数
# .view(-1)将输出展平为一维张量
# .float()将数据类型转换为浮点型
with torch.no_grad():
    scores = model(**inputs).logits.view(-1).float()

# 打印模型预测的相关性分数
# 分数越高表示查询和文档的相关性越强
print(scores)  # 输出相关性分数

pairs = [
    ['what is panda?', 'The giant panda is a bear species endemic to China.'],  # 高相关
    ['what is panda?', 'Pandas are cute.'],                                     # 中等相关
    ['what is panda?', 'The Eiffel Tower is in Paris.']                        # 不相关
]
inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt')
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    scores = model(**inputs).logits.view(-1).float()
print(scores)  # 输出相关性分数

