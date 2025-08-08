# 模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('BAAI/bge-m3', cache_dir='/root/autodl-tmp/models')


from FlagEmbedding import BGEM3FlagModel
import torch
# 检查是否有可用的GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")
model = BGEM3FlagModel('/root/autodl-tmp/models/BAAI/bge-m3',  
                       use_fp16=True, # 设置use_fp16为True可以加快计算速度，但会略微降低性能
                       device=device, # 模型将在指定的设备上运行，这里是GPU
                       local_files_only=True, # 仅从本地文件加载模型，不尝试下载，默认False
                       ) 
# 如果需要查看GPU信息
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")


sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

# 使用BGE-M3模型对第一组句子进行编码
# batch_size=12 表示每批处理12个句子，可以根据GPU内存大小调整
# max_length=8192 表示输入文本的最大长度限制
# dense_vecs 获取密集向量表示，用于后续相似度计算
embeddings_1 = model.encode(sentences_1, 
                            batch_size=12, 
                            max_length=8192, # 如果不需要这么长的长度，可以设置一个较小的值来加快编码过程
                            )['dense_vecs']
# 使用BGE-M3模型对第二组句子进行编码
# 这里使用了默认参数:
# - batch_size默认为12
# - max_length默认为8192
# dense_vecs用于获取密集向量表示
embeddings_2 = model.encode(sentences_2)['dense_vecs']
# @ 符号在Python中表示矩阵乘法运算
# => 通过矩阵乘法计算了两组句子之间的余弦相似度矩阵。结果 similarity 的形状是 [sentences_1的数量, sentences_2的数量]。
# 计算两组句子嵌入向量之间的余弦相似度
# embeddings_1: 第一组句子的嵌入向量矩阵，形状为 [sentences_1数量, 向量维度]
# embeddings_2.T: 第二组句子的嵌入向量矩阵的转置，形状为 [向量维度, sentences_2数量]
# @ 运算符执行矩阵乘法，得到相似度矩阵，形状为 [sentences_1数量, sentences_2数量]
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
# [[0.6265, 0.3477], [0.3499, 0.678 ]]

