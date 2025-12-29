import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
import src.prompts as prompts
from concurrent.futures import ThreadPoolExecutor


# JinaReranker：基于Jina API的重排器，适用于多语言场景
class JinaReranker:
    def __init__(self):
        # 初始化Jina重排API地址和请求头
        self.url = 'https://api.jina.ai/v1/rerank'
        self.headers = self.get_headers()
        
    def get_headers(self):
        # 加载Jina API密钥，组装请求头
        load_dotenv()
        jina_api_key = os.getenv("JINA_API_KEY")    
        headers = {'Content-Type': 'application/json',
                   'Authorization': f'Bearer {jina_api_key}'}
        return headers
    
    def rerank(self, query, documents, top_n = 10):
        # 调用Jina API进行重排，返回top_n相关文档
        data = {
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "top_n": top_n,
            "documents": documents
        }

        response = requests.post(url=self.url, headers=self.headers, json=data)

        return response.json()

# LLMReranker：基于大模型的重排器，支持单条和批量重排
class LLMReranker:
    def __init__(self, provider: str = "dashscope"):
        # 支持 openai/dashscope，默认 dashscope
        self.provider = provider.lower()
        self.llm = self.set_up_llm()
        self.system_prompt_rerank_single_block = prompts.RerankingPrompt.system_prompt_rerank_single_block
        self.system_prompt_rerank_multiple_blocks = prompts.RerankingPrompt.system_prompt_rerank_multiple_blocks
        self.schema_for_single_block = prompts.RetrievalRankingSingleBlock
        self.schema_for_multiple_blocks = prompts.RetrievalRankingMultipleBlocks
      
    def set_up_llm(self):
        # 根据 provider 初始化 LLM 客户端
        load_dotenv()
        if self.provider == "openai":
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == "dashscope":
            import dashscope
            dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
            return dashscope
        else:
            raise ValueError(f"不支持的 LLM provider: {self.provider}")
    
    def get_rank_for_single_block(self, query, retrieved_document):
        # 针对单个文本块，调用LLM进行相关性评分
        user_prompt = f'/nHere is the query:/n"{query}"/n/nHere is the retrieved text block:/n"""/n{retrieved_document}/n"""/n'
        if self.provider == "openai":
            completion = self.llm.beta.chat.completions.parse(
                model="gpt-4o-mini-2024-07-18",
                temperature=0,
                messages=[
                    {"role": "system", "content": self.system_prompt_rerank_single_block},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=self.schema_for_single_block
            )
            response = completion.choices[0].message.parsed
            response_dict = response.model_dump()
            return response_dict
        elif self.provider == "dashscope":
            # dashscope 只返回字符串，暂不做结构化解析
            messages = [
                {"role": "system", "content": self.system_prompt_rerank_single_block},
                {"role": "user", "content": user_prompt},
            ]
            rsp = self.llm.Generation.call(
                model="qwen-turbo",
                messages=messages,
                temperature=0,
                result_format='message'
            )
            # 健壮性检查，防止 rsp 为 None 或非 dict
            if not rsp or not isinstance(rsp, dict):
                raise RuntimeError(f"DashScope返回None或非dict: {rsp}")
            if 'output' in rsp and 'choices' in rsp['output']:
                content = rsp['output']['choices'][0]['message']['content']
                # 这里只返回字符串，后续可按需解析
                return {"relevance_score": 0.0, "reasoning": content}
            else:
                raise RuntimeError(f"DashScope返回格式异常: {rsp}")
        else:
            raise ValueError(f"不支持的 LLM provider: {self.provider}")

    def get_rank_for_multiple_blocks(self, query, retrieved_documents):
        # 针对多个文本块，批量调用LLM进行相关性评分
        formatted_blocks = "\n\n---\n\n".join([f'Block {i+1}:\n\n"""\n{text}\n"""' for i, text in enumerate(retrieved_documents)])
        user_prompt = (
            f"Here is the query: \"{query}\"\n\n"
            "Here are the retrieved text blocks:\n"
            f"{formatted_blocks}\n\n"
            f"You should provide exactly {len(retrieved_documents)} rankings, in order."
        )
        if self.provider == "openai":
            completion = self.llm.beta.chat.completions.parse(
                model="gpt-4o-mini-2024-07-18",
                temperature=0,
                messages=[
                    {"role": "system", "content": self.system_prompt_rerank_multiple_blocks},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=self.schema_for_multiple_blocks
            )
            response = completion.choices[0].message.parsed
            response_dict = response.model_dump()
            return response_dict
        elif self.provider == "dashscope":
            messages = [
                {"role": "system", "content": self.system_prompt_rerank_multiple_blocks},
                {"role": "user", "content": user_prompt},
            ]
            rsp = self.llm.Generation.call(
                model="qwen-turbo",
                messages=messages,
                temperature=0,
                result_format='message'
            )
            # 健壮性检查，防止 rsp 为 None 或非 dict
            if not rsp or not isinstance(rsp, dict):
                raise RuntimeError(f"DashScope返回None或非dict: {rsp}")
            #print('rsp=', rsp)
            if 'output' in rsp and 'choices' in rsp['output']:
                content = rsp['output']['choices'][0]['message']['content']
                # 这里只返回字符串，后续可按需解析
                return {"block_rankings": [{"relevance_score": 0.0, "reasoning": content} for _ in retrieved_documents]}
            else:
                raise RuntimeError(f"DashScope返回格式异常: {rsp}")
        else:
            raise ValueError(f"不支持的 LLM provider: {self.provider}")

    def rerank_documents(self, query: str, documents: list, documents_batch_size: int = 4, llm_weight: float = 0.7):
        """
        使用多线程并行方式对多个文档进行重排。
        结合向量相似度和LLM相关性分数，采用加权平均融合。
        参数：
            query: 查询语句
            documents: 待重排的文档列表，每个元素需包含'text'和'distance'
            documents_batch_size: 每批送入LLM的文档数
            llm_weight: LLM分数权重（0-1），其余为向量分数权重
        返回：
            按融合分数降序排序的文档列表
        """
        # 按batch分组
        doc_batches = [documents[i:i + documents_batch_size] for i in range(0, len(documents), documents_batch_size)]
        vector_weight = 1 - llm_weight
        
        if documents_batch_size == 1:
            def process_single_doc(doc):
                # 单文档重排
                ranking = self.get_rank_for_single_block(query, doc['text'])
                
                doc_with_score = doc.copy()
                doc_with_score["relevance_score"] = ranking["relevance_score"]
                # 计算融合分数，distance越小越相关
                doc_with_score["combined_score"] = round(
                    llm_weight * ranking["relevance_score"] + 
                    vector_weight * doc['distance'],
                    4
                )
                return doc_with_score

            # 多线程并行处理，max_workers=1 保证 dashscope LLM 串行调用，避免 QPS 超限
            with ThreadPoolExecutor(max_workers=1) as executor:
                all_results = list(executor.map(process_single_doc, documents))
                
        else:
            def process_batch(batch):
                # 批量重排
                texts = [doc['text'] for doc in batch]
                rankings = self.get_rank_for_multiple_blocks(query, texts)
                results = []
                block_rankings = rankings.get('block_rankings', [])
                
                if len(block_rankings) < len(batch):
                    print(f"\nWarning: Expected {len(batch)} rankings but got {len(block_rankings)}")
                    for i in range(len(block_rankings), len(batch)):
                        doc = batch[i]
                        print(f"Missing ranking for document on page {doc.get('page', 'unknown')}:")
                        print(f"Text preview: {doc['text'][:100]}...\n")
                    
                    for _ in range(len(batch) - len(block_rankings)):
                        block_rankings.append({
                            "relevance_score": 0.0, 
                            "reasoning": "Default ranking due to missing LLM response"
                        })
                
                for doc, rank in zip(batch, block_rankings):
                    doc_with_score = doc.copy()
                    doc_with_score["relevance_score"] = rank["relevance_score"]
                    doc_with_score["combined_score"] = round(
                        llm_weight * rank["relevance_score"] + 
                        vector_weight * doc['distance'],
                        4
                    )
                    results.append(doc_with_score)
                return results

            # 多线程并行处理，max_workers=1 保证 dashscope LLM 串行调用，避免 QPS 超限
            with ThreadPoolExecutor(max_workers=1) as executor:
                batch_results = list(executor.map(process_batch, doc_batches))
            
            # 扁平化结果
            all_results = []
            for batch in batch_results:
                all_results.extend(batch)
        
        # 按融合分数降序排序
        all_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return all_results
