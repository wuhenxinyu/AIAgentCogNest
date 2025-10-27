"""
Gensim工具:
    • 可以从非结构化文本中，无监督地学习到隐层的主题向量表达
    • 每一个向量变换的操作都对应着一个主题模型
    • 支持TF-IDF，LDA, LSA, word2vec等多种主题模型算法


Word2Vec工具的使用：
    • Word Embedding就是将Word嵌入到一个数学空间里，Word2vec，就是词嵌入的一种
    • 可以将sentence中的word转换为固定大小的向量表达（Vector Respresentations），
    • 其中意义相近的词将被映射到向量空间中相近的位置。
    • 将待解决的问题转换成为单词word和文章doc的对应关系，从而可以使用向量空间中的距离来度量文章之间的相似度。
    • 大V推荐中，大V => 单词，将每一个用户关注大V的顺序 => 文章
    • 商品推荐中，商品 => 单词，用户对商品的行为顺序 => 文章    

场景：计算小说中的人物相似度，比如孙悟空与猪八戒，孙悟空与孙行者
方案步骤：
• Step1，使用分词工具进行分词，比如NLTK,JIEBA
• Step2，将训练语料转化成一个sentence的迭代器
• Step3，使用word2vec进行训练
• Step4，计算两个单词的相似度
"""
# 先运行 word_seg进行中文分词，然后再进行word_similarity计算
# 将Word转换成Vec，然后计算相似度 
from gensim.models import word2vec
import multiprocessing

# 如果目录中有多个文件，可以使用PathLineSentences
segment_folder = '12-Embeddings和向量数据库/word2vec/data/journey_to_the_west/segment'

# 使用 gensim 的 PathLineSentences 类从指定文件夹中读取已分词的文本数据
# 该类会将文件夹内每个文件的每一行视为一个句子，最终得到一个句子的迭代器
# segment_folder 是之前定义的包含分词后文本文件的文件夹路径
sentences = word2vec.PathLineSentences(segment_folder)

# 创建第一个 Word2Vec 模型并进行训练
# sentences 是输入的句子迭代器，用于模型训练
# vector_size=100 表示每个词将被转换为一个 100 维的向量
# window=3 表示训练时上下文窗口的大小为 3，即预测一个词时会考虑其前后各 3 个词
# min_count=1 表示词频小于 1 的词将被忽略，这里设为 1 意味着保留所有词
model = word2vec.Word2Vec(sentences, vector_size=100, window=3, min_count=1)

# 打印 '孙悟空' 对应的词向量，当前代码行被注释，若需要查看可取消注释
print(model.wv['孙悟空'])

# 计算并打印 '孙悟空' 和 '猪八戒' 的相似度
# 使用模型的 wv 属性访问词向量相关功能，similarity 方法返回两个词的余弦相似度
print(model.wv.similarity('孙悟空', '猪八戒'))

# 计算并打印 '孙悟空' 和 '孙行者' 的相似度
print(model.wv.similarity('孙悟空', '孙行者'))

# 计算并打印与 '孙悟空' 和 '唐僧' 相加，再减去 '孙行者' 最相似的词
# most_similar 方法可以实现词向量的加减运算，positive 参数指定相加的词，negative 参数指定相减的词
# 返回一个列表，包含最相似的若干词及其相似度得分
print(model.wv.most_similar(positive=['孙悟空', '唐僧'], negative=['孙行者']))

# 创建第二个 Word2Vec 模型并进行训练
# 与第一个模型不同，这里 vector_size 设置为 128，即每个词将被转换为 128 维的向量
# window 设置为 5，扩大了上下文窗口的大小
# min_count 设置为 5，意味着词频小于 5 的词将被忽略
# workers=multiprocessing.cpu_count() 表示使用当前机器的所有 CPU 核心并行训练模型，提高训练速度
model2 = word2vec.Word2Vec(sentences, vector_size=128, window=5, min_count=5, workers=multiprocessing.cpu_count())

# 将训练好的第二个模型保存到指定路径
# 后续可以使用 word2vec.Word2Vec.load() 方法加载该模型
model2.save('12-Embeddings和向量数据库/word2vec/models/word2Vec.model')

# 使用第二个模型计算并打印 '孙悟空' 和 '猪八戒' 的相似度
print(model2.wv.similarity('孙悟空', '猪八戒'))

# 使用第二个模型计算并打印 '孙悟空' 和 '孙行者' 的相似度
print(model2.wv.similarity('孙悟空', '孙行者'))

# 使用第二个模型计算并打印与 '孙悟空' 和 '唐僧' 相加，再减去 '孙行者' 最相似的词
print(model2.wv.most_similar(positive=['孙悟空', '唐僧'], negative=['孙行者']))