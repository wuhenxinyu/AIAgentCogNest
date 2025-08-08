

#模型下载
from modelscope import snapshot_download
#model_dir = snapshot_download('iic/gte_Qwen2-7B-instruct', cache_dir='/root/autodl-tmp/models')
model_dir = snapshot_download('iic/gte_Qwen2-1.5B-instruct', cache_dir='/root/autodl-tmp/models')


from sentence_transformers import SentenceTransformer
import torch
# 检查是否有可用的GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")
model_dir = "/root/autodl-tmp/models/iic/gte_Qwen2-1.5B-instruct"
# 初始化SentenceTransformer模型
# model_dir: 模型所在的本地目录路径
# trust_remote_code=True: 允许加载和执行远程代码，这在加载一些自定义模型时是必要的
model = SentenceTransformer(model_dir, trust_remote_code=True, device=device)

# 如果需要查看GPU信息
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")


# 设置模型的最大序列长度为8192个token
# 这个设置决定了模型能处理的最长文本长度
# 超过此长度的文本会被截断
model.max_seq_length = 8192

queries = [
    "how much protein should a female eat",
    "summit define",
]
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
]

# 使用模型将查询文本转换为向量表示，prompt_name="query"用于指定使用查询模式进行编码
query_embeddings = model.encode(queries, prompt_name="query")

# 使用模型将文档文本转换为向量表示
document_embeddings = model.encode(documents)

# 计算查询向量和文档向量之间的相似度分数
# @ 操作符用于矩阵乘法，.T进行矩阵转置
# 将结果乘以100以获得百分比形式的相似度分数
scores = (query_embeddings @ document_embeddings.T) * 100
print(scores.tolist())
# [[70.00668334960938, 8.184843063354492], [14.62419319152832, 77.71407318115234]]

