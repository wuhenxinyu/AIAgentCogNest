"""
酒店推荐系统
余弦相似度：
    • 通过测量两个向量的夹角的余弦值来度量它们之间的相似性。
    • 判断两个向量大致方向是否相同，方向相同时，余弦相似度为1；两个向量夹角为90°时，余弦相似度的值为0，方向完全相反时，余弦相似度的值为-1。
    • 两个向量之间夹角的余弦值为[-1, 1]，值越接近1，两个向量的方向越接近；值越接近-1，两个向量的方向越相反。
    • 余弦相似度的计算方法：
        余弦相似度 = 向量A · 向量B / (||向量A|| * ||向量B||)
        其中，向量A · 向量B表示向量A和向量B的点积，||向量A||和||向量B||表示向量A和向量B的范数（即向量的长度）。

什么是N-Gram（N元语法）：
    • 基于一个假设：第n个词出现与前n-1个词相关，而与其他任何词不相关.
    • N=1时为unigram，N=2为bigram，N=3为trigram
    • N-Gram指的是给定一段文本，其中的N个item的序列,比如文本：A B C D E，对应的Bi-Gram为A B, B C, C D, D E
    • 当一阶特征不够用时，可以用N-Gram做为新的特征。比如在处理文本特征时，一个关键词是一个特征，但有些情况不够用，需要提取更多的特征，采用N-Gram => 可以理解是相邻两个关键词的特征组合

N-Gram + TF-IDF的特征表达会让特征矩阵非常系数，计算量大，有没有更适合的方式？ Embedding

"""

import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
pd.options.display.max_columns = 30
import matplotlib.pyplot as plt
# 支持中文
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
df = pd.read_csv('/Users/clz/Downloads/xmkf/llm/AIAgentCogNest/12-Embeddings和向量数据库/hotel_recommendation/Seattle_Hotels.csv', encoding="latin-1")

# 数据探索
print(df.head())
print('数据集中的酒店个数：', len(df))

# 创建英文停用词列表
ENGLISH_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', 
    "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 
    'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 
    'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 
    "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', 
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', 
    "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

def print_description(index):
    example = df[df.index == index][['desc', 'name']].values[0]
    if len(example) > 0:
        print(example[0])
        print('Name:', example[1])
print('第10个酒店的描述：')
print_description(10)



# 得到酒店描述中n-gram特征中的TopK个
def get_top_n_words(corpus, n=1, k=None):
    # 统计ngram词频矩阵，使用自定义停用词列表
    """
    CountVectorizer：
        • 将文本中的词语转换为词频矩阵
        • fit_transform：计算各个词语出现的次数
        • get_feature_names_out ： 可获得所有文本的关键词
        • toarray()：查看词频矩阵的结果。
    """
    vec = CountVectorizer(ngram_range=(n, n), stop_words=list(ENGLISH_STOPWORDS)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    """
    print('feature names:')
    print(vec.get_feature_names_out())
    print('bag of words:')
    print(bag_of_words.toarray())
    """
    sum_words = bag_of_words.sum(axis=0)
    # 遍历词表，将每个词与其对应的词频组合成元组，构建词频列表
    # vec.vocabulary_ 是一个字典，键为词语，值为该词语在词频矩阵中的索引
    # sum_words[0, idx] 表示该词语在语料库中的总词频
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    # 按照词频从大到小对词频列表进行排序
    # key = lambda x: x[1] 表示以元组中的第二个元素（即词频）作为排序依据
    # reverse=True 表示降序排序，即词频高的词排在前面
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:k]

    
common_words = get_top_n_words(df['desc'], n=3, k=20)
#print(common_words)
df1 = pd.DataFrame(common_words, columns = ['desc' , 'count'])
df1.groupby('desc').sum()['count'].sort_values().plot(kind='barh', title='去掉停用词后，酒店描述中的Top20单词')
plt.show()


# 文本预处理
REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
# 使用自定义的英文停用词列表替代nltk的stopwords
STOPWORDS = ENGLISH_STOPWORDS
# 对文本进行清洗
def clean_text(text):
    # 全部小写
    text = text.lower()
    # 用空格替代一些特殊符号，如标点
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    # 移除BAD_SYMBOLS_RE
    text = BAD_SYMBOLS_RE.sub('', text)
    # 从文本中去掉停用词
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) 
    return text
# 对desc字段进行清理，apply针对某列
df['desc_clean'] = df['desc'].apply(clean_text)
#print(df['desc_clean'])


# 建模
df.set_index('name', inplace = True)
"""
# TF-IDF（Term Frequency-Inverse Document Frequency）即词频-逆文档频率，是一种用于信息检索与文本挖掘的常用加权技术。
# 词频（TF）表示某词项在文档中出现的频率，体现了该词项对文档的重要性；逆文档频率（IDF）衡量了一个词项的普遍重要性，
# 如果一个词项在很多文档中都出现，其IDF值较低，意味着该词项区分度较低；反之IDF值较高，区分度较高。
# TF-IDF的作用是评估一个词项在文档集合中的重要性，通过计算TF和IDF的乘积得到词项的权重，权重越高表示该词项对文档的代表性越强。
# 此处使用TF-IDF提取文本特征，并使用自定义停用词列表，以过滤掉常见无实际意义的词汇，提高特征质量。
"""
# 使用TF-IDF提取文本特征
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0.01, stop_words=list(ENGLISH_STOPWORDS))
# 针对desc_clean提取tfidf
tfidf_matrix = tf.fit_transform(df['desc_clean'])
print('TFIDF feature names:')
#print(tf.get_feature_names_out())
print(len(tf.get_feature_names_out()))
print('tfidf_matrix:')
print(tfidf_matrix)
print(tfidf_matrix.shape)
# 计算酒店之间的余弦相似度（线性核函数）
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
#print(cosine_similarities)
print(cosine_similarities.shape)
# 此代码行的作用是将DataFrame `df` 的索引转换为一个 Pandas 的 Series 对象。
# 由于 `df` 的索引已在之前设置为酒店名称（`df.set_index('name', inplace = True)`），
# 因此这里创建的 `indices` Series 包含了数据集中所有酒店的名称。
# 该 Series 后续可用于根据酒店名称查找对应的索引位置，方便在相似度矩阵中定位特定酒店的相似度数据。
indices = pd.Series(df.index) # df.index 是酒店名称

# 基于相似度矩阵和指定的酒店name，推荐TOP10酒店
def recommendations(name, cosine_similarities = cosine_similarities):
    recommended_hotels = []
    # 找到想要查询酒店名称的idx
    idx = indices[indices == name].index[0]
    print('idx=', idx)
    # 对于idx酒店的余弦相似度向量按照从大到小进行排序
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending = False)
    # 取相似度最大的前10个（除了自己以外）
    top_10_indexes = list(score_series.iloc[1:11].index)
    # 放到推荐列表中
    for i in top_10_indexes:
        recommended_hotels.append(list(df.index)[i])
    return recommended_hotels


print(recommendations('Hilton Seattle Airport & Conference Center'))
print(recommendations('The Bacon Mansion Bed and Breakfast'))
#print(result)
