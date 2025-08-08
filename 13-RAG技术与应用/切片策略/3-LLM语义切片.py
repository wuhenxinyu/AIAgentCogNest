from openai import OpenAI
import json
import os
import re

def advanced_semantic_chunking_with_llm(text, max_chunk_size=512):
    """使用LLM进行高级语义切片"""
    os.environ["DASHSCOPE_API_KEY"] = "your_api_key_here"


    # 检查环境变量
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("警告: 未设置 DASHSCOPE_API_KEY 环境变量，将使用基础语义切片")
        return fallback_semantic_chunking(text, max_chunk_size)
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.fe8.cn/v1"
    )
    
    prompt = f"""
请将以下文本按照语义完整性进行切片，每个切片不超过{max_chunk_size}字符。
要求：
1. 保持语义完整性
2. 在自然的分割点切分
3. 返回JSON格式的切片列表，格式如下：
{{
  "chunks": [
    "第一个切片内容",
    "第二个切片内容",
    ...
  ]
}}

文本内容：
{text}

请返回JSON格式的切片列表：
"""
    
    try:
        print("正在调用LLM进行语义切片...")
        response = client.chat.completions.create(
            model="qwen-turbo-latest",
            messages=[
                {"role": "system", "content": "你是一个专业的文本切片助手。请严格按照JSON格式返回结果，不要添加任何额外的标记。"},
                {"role": "user", "content": prompt}
            ]
        )
        
        result = response.choices[0].message.content
        # 打印LLM返回的前200个字符,使用省略号表示后续内容
        print(f"LLM返回结果: {result[:200]}...")
        
        # 清理结果，移除可能的Markdown代码块标记
        cleaned_result = result.strip()
        if cleaned_result.startswith('```'):
            # 移除开头的 ```json 或 ```
            cleaned_result = re.sub(r'^```(?:json)?\s*', '', cleaned_result)
        if cleaned_result.endswith('```'):
            # 移除结尾的 ```
            cleaned_result = re.sub(r'\s*```$', '', cleaned_result)
        
        # 解析JSON结果
        chunks_data = json.loads(cleaned_result)
        
        # 处理不同的返回格式
        if "chunks" in chunks_data:
            return chunks_data["chunks"]
        elif "slice" in chunks_data:
            # 如果返回的是包含"slice"字段的列表
            if isinstance(chunks_data, list):
                return [item.get("slice", "") for item in chunks_data if item.get("slice")]
            else:
                return [chunks_data["slice"]]
        else:
            # 如果直接返回字符串列表
            if isinstance(chunks_data, list):
                return chunks_data
            else:
                print(f"意外的返回格式: {chunks_data}")
                return []
        
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {e}")
        print(f"原始结果: {result}")
        # 尝试手动解析
        try:
            # 尝试提取JSON部分
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                chunks_data = json.loads(json_str)
                if "chunks" in chunks_data:
                    return chunks_data["chunks"]
        except:
            pass        
    except Exception as e:
        print(f"LLM切片失败: {e}")

def test_chunking_methods():
    """测试不同的切片方法"""
    # 示例文本
    text = """
迪士尼乐园提供多种门票类型以满足不同游客需求。一日票是最基础的门票类型，可在购买时选定日期使用，价格根据季节浮动。两日票需要连续两天使用，总价比购买两天单日票优惠约9折。特定日票包含部分节庆活动时段，需注意门票标注的有效期限。

购票渠道以官方渠道为主，包括上海迪士尼官网、官方App、微信公众号及小程序。第三方平台如飞猪、携程等合作代理商也可购票，但需认准官方授权标识。所有电子票需绑定身份证件，港澳台居民可用通行证，外籍游客用护照，儿童票需提供出生证明或户口本复印件。

生日福利需在官方渠道登记，可获赠生日徽章和甜品券。半年内有效结婚证持有者可购买特别套票，含皇家宴会厅双人餐。军人优惠现役及退役军人凭证件享8折，需至少提前3天登记审批。
"""

    print("\n=== LLM高级语义切片测试 ===")
    try:
        chunks = advanced_semantic_chunking_with_llm(text, max_chunk_size=300)
        print(f"LLM高级语义切片生成 {len(chunks)} 个切片:")
        # 使用enumerate遍历chunks列表,获取每个切片的索引和内容
        for i, chunk in enumerate(chunks):
            print(f"LLM语义块 {i+1} (长度: {len(chunk)}): {chunk}")
    except Exception as e:
        print(f"LLM切片测试失败: {e}")

if __name__ == "__main__":
    test_chunking_methods()