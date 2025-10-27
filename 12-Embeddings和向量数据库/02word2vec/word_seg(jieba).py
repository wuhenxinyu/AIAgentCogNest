"""
中文分词，jieba分词并过滤停用词
"""

# 对txt文件进行中文分词
import warnings
# 忽略pkg_resources弃用警告
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

import jieba
import os
from utils import files_processing

# 源文件所在目录
source_folder = '12-Embeddings和向量数据库/word2vec/data/journey_to_the_west/source'
segment_folder = '12-Embeddings和向量数据库/word2vec/data/journey_to_the_west/segment'


# 字词分割，对整个文件内容进行字词分割
def segment_lines(file_list,segment_out_dir,stopwords=[]):
    """
    对文件列表中的每个文件进行中文分词，并过滤停用词，最后将分词结果写入新文件。

    参数:
    file_list (list): 待处理的文件路径列表
    segment_out_dir (str): 分词结果文件的输出目录
    stopwords (list): 停用词列表，默认为空列表

    返回:
    无，直接将分词结果写入输出文件
    """
    # 遍历文件列表，使用enumerate同时获取文件索引和文件路径
    for i,file in enumerate(file_list):
        # 构建输出文件的路径，文件名格式为 segment_索引.txt
        segment_out_name=os.path.join(segment_out_dir,'segment_{}.txt'.format(i))
        # 以二进制只读模式打开当前文件
        with open(file, 'rb') as f:
            # 读取文件的全部内容
            document = f.read()
            # 使用jieba库对文件内容进行中文分词
            document_cut = jieba.cut(document)
            # 初始化一个空列表，用于存储过滤停用词后的分词结果
            sentence_segment=[]
            # 遍历分词结果
            for word in document_cut:
                # 检查当前词是否不在停用词列表中
                if word not in stopwords:
                    # 若不在停用词列表中，则将该词添加到结果列表中
                    sentence_segment.append(word)
            # 将分词结果列表用空格连接成字符串
            result = ' '.join(sentence_segment)
            # 将结果字符串编码为UTF-8字节流
            result = result.encode('utf-8')
            # 以二进制写入模式打开输出文件
            with open(segment_out_name, 'wb') as f2:
                # 将编码后的结果写入输出文件
                f2.write(result)

# 对source中的txt文件进行分词，输出到segment目录中
file_list=files_processing.get_files_list(source_folder, postfix='*.txt')

segment_lines(file_list, segment_folder)
