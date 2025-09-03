"""
下载MinerU模型,MinerU是多个小模型组合,每个小模型都有自己的模型文件和配置文件,Qweb-VL是大模型。
这个脚本会下载所有的模型文件和配置文件,并保存到指定的目录下
Step1，安装magic-pdf，pip install -U "magic-pdf[full]"
Step2，安装modelscope，pip install modelscope
Step3，运行这个脚本，python 3_download_models(modelscope).py,下载依赖的models
Step4，推理pdf，magic-pdf -p 三国演义.pdf -o ./output 

作用场景： 1、企业对文档做知识库,企业可以上传文档,文档会被解析成图片,图片会被上传到 MinerU 模型,模型会返回图片的特征向量,特征向量会被保存到数据库中,企业可以根据特征向量进行文档检索。
         2、对网页抓取,企业可以上传网页,网页会被解析成图片,图片会被上传到 MinerU 模型,模型会返回图片的特征向量,特征向量会被保存到数据库中,企业可以根据特征向量进行网页检索。
         3、对图片抓取,企业可以上传图片,图片会被上传到 MinerU 模型,模型会返回图片的特征向量,特征向量会被保存到数据库中,企业可以根据特征向量进行图片检索。

使用的模型：
    1、LayoutLMv3 ：统一的文本和图像掩码来预训练文档 AI 的多模态 Transformer
    2、YOLO ：检测图片中的物体，布局检测，目标检测的深度学习算法，通过卷积神经网络直接预测图像中的目标位置和类别
    3、unimernet_hf_small_2503 ：数学公式识别模型
    4、paddleocr_torch ：OCR识别模型，核心作用是将图片中的文字转换为可编辑的文本格式，支持多任务处理（如文字检测、方向分类、文本识别等），并优化了模型加载和推理速度。
"""
import json
import shutil
import os

import requests
from modelscope import snapshot_download


def download_json(url):
    # 下载JSON文件
    response = requests.get(url)
    response.raise_for_status()  # 检查请求是否成功
    return response.json()


def download_and_modify_json(url, local_filename, modifications):
    if os.path.exists(local_filename):
        data = json.load(open(local_filename))
        config_version = data.get('config_version', '0.0.0')
        if config_version < '1.3.0':
            data = download_json(url)
    else:
        data = download_json(url)

    # 修改内容
    for key, value in modifications.items():
        data[key] = value

    # 保存修改后的内容
    with open(local_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    mineru_patterns = [
        # "models/Layout/LayoutLMv3/*",
        "models/Layout/YOLO/*",
        "models/MFD/YOLO/*",
        "models/MFR/unimernet_hf_small_2503/*",
        "models/OCR/paddleocr_torch/*",
        # "models/TabRec/TableMaster/*",
        # "models/TabRec/StructEqTable/*",
    ]
    # 指定模型保存目录为当前目录下的 'modelscope_models' 文件夹，避免保存到 C 盘
    local_model_dir = os.path.join(os.getcwd(), 'modelscope_models')
    model_dir = snapshot_download('opendatalab/PDF-Extract-Kit-1.0', allow_patterns=mineru_patterns, local_dir=local_model_dir)
    layoutreader_model_dir = snapshot_download('ppaanngggg/layoutreader', local_dir=local_model_dir)
    model_dir = model_dir + '/models'
    print(f'model_dir is: {model_dir}')
    print(f'layoutreader_model_dir is: {layoutreader_model_dir}')

    # paddleocr_model_dir = model_dir + '/OCR/paddleocr'
    # user_paddleocr_dir = os.path.expanduser('~/.paddleocr')
    # if os.path.exists(user_paddleocr_dir):
    #     shutil.rmtree(user_paddleocr_dir)
    # shutil.copytree(paddleocr_model_dir, user_paddleocr_dir)

    json_url = 'https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/mineru.template.json'
    config_file_name = 'mineru.json'
    home_dir = os.path.expanduser('~')
    config_file = os.path.join(home_dir, config_file_name)

    json_mods = {
        'models-dir': model_dir,
        'layoutreader-model-dir': layoutreader_model_dir,
    }

    download_and_modify_json(json_url, config_file, json_mods)
    print(f'The configuration file has been configured successfully, the path is: {config_file}')
