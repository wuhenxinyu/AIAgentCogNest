# qwen_agent/searcher/elasticsearch_searcher.py
import os
import json
import hashlib
import logging
from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import BadRequestError # 导入特定的异常
from qwen_agent.tools.doc_parser import DocParser
from openai import OpenAI

# 为此模块设置一个日志记录器
logger = logging.getLogger(__name__)


class ElasticsearchSearcher:
    """一个使用 Elasticsearch 进行文档索引和搜索的搜索器。"""

    def __init__(self, cfg):
        self.cfg = cfg
        es_cfg = cfg.get('es', {})
        self.host = es_cfg.get('host', 'http://localhost')
        self.port = es_cfg.get('port', 9200)
        self.user = es_cfg.get('user')
        self.password = es_cfg.get('password')
        self.index_name = es_cfg.get('index_name', 'qwen_agent_rag_idx')
        self.use_embedding = es_cfg.get('use_embedding', False)
        self.embedding_model = es_cfg.get('embedding_model', 'text-embedding-v4')
        self.embedding_dimensions = es_cfg.get('embedding_dimensions', 1024)
        self.similarity_threshold = es_cfg.get('similarity_threshold', 0.5)
        # 从环境变量获取 API Key，如果未设置则使用默认值（注意：生产环境中应避免硬编码）
        api_key = os.getenv('DASHSCOPE_API_KEY', 'sk-58f051ae745e4bb19fdca31735105b11')
        os.environ['DASHSCOPE_API_KEY'] = api_key
        # DocParser 用于解析和分块文档
        self.parser = DocParser(cfg=self.cfg)
        
        # 初始化 embedding 客户端（如果启用 embedding 模式）
        if self.use_embedding:
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                logger.warning("DASHSCOPE_API_KEY 环境变量未设置，无法初始化 embedding 客户端。将回退到普通搜索模式。")
                self.use_embedding = False  # 回退到普通模式
            else:
                try:
                    self.embedding_client = OpenAI(
                        api_key=api_key,
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                    )
                    logger.info("成功初始化 embedding 客户端。")
                except Exception as e:
                    logger.error(f"初始化 embedding 客户端时发生错误: {e}")
                    logger.error("将回退到普通搜索模式。")
                    self.use_embedding = False  # 回退到普通模式
        
        self.client = self._connect()
        if self.client:
            logger.info("成功连接到 Elasticsearch！")
            self._create_index_if_not_exists()
        else:
            logger.error("连接 Elasticsearch 失败。请检查您的配置、网络和 ES 服务状态。")

    def _connect(self) -> Elasticsearch:
        """建立并返回到 Elasticsearch 的连接。"""
        try:
            # 根据提供的配置构建连接参数
            es_args = {
                'hosts': [{
                    'host': self.host.replace('https://', '').replace('http://', ''),
                    'port': self.port,
                    'scheme': 'https' if 'https' in self.host else 'http',
                }],
                'verify_certs': False, # 在生产环境中应设为 True 并提供证书
                'request_timeout': 60,
            }
            if self.user and self.password:
                es_args['basic_auth'] = (self.user, self.password)

            client = Elasticsearch(**es_args)
            
            # 检查连接
            if not client.ping():
                raise ConnectionError("Elasticsearch ping 失败。")
            
            return client
        except Exception as e:
            logger.error(f"无法连接到 Elasticsearch：{e}")
            return None

    def _create_index_if_not_exists(self):
        """
        如果索引不存在，则创建它。
        根据是否使用 embedding 模式创建不同类型的索引。
        """
        try:
            if not self.client.indices.exists(index=self.index_name):
                logger.info(f"索引 '{self.index_name}' 不存在，正在创建...")
                
                if self.use_embedding:
                    # 创建支持向量搜索的索引
                    embedding_index_settings = {
                        "mappings": {
                            "properties": {
                                "content": {"type": "text"},
                                "source": {"type": "keyword"},
                                "content_vector": {
                                    "type": "dense_vector",
                                    "dims": self.embedding_dimensions,
                                    "index": "true",
                                    "similarity": "cosine"
                                }
                            }
                        }
                    }
                    
                    try:
                        self.client.indices.create(index=self.index_name, body=embedding_index_settings)
                        logger.info(f"成功创建支持向量搜索的索引 '{self.index_name}' (维度: {self.embedding_dimensions})。")
                    except BadRequestError as e:
                        logger.error(f"创建向量索引时发生错误: {e}")
                        raise e
                else:
                    # 优先尝试使用 IK 分词器的配置（BM25搜索）
                    ik_index_settings = {
                        "settings": {"analysis": {"analyzer": {"default": {"type": "ik_max_word"}}}},
                        "mappings": {
                            "properties": {
                                "content": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                                "source": {"type": "keyword"}
                            }
                        }
                    }
                    
                    try:
                        # 首次尝试使用 IK 创建
                        self.client.indices.create(index=self.index_name, body=ik_index_settings)
                        logger.info(f"成功使用 IK 分词器创建索引 '{self.index_name}'。")
                    except BadRequestError as e:
                        # 捕获因分词器不存在导致的错误
                        if 'Unknown analyzer type [ik_max_word]' in str(e):
                            logger.warning("未能找到 'ik_max_word' 分词器。这通常是因为 Elasticsearch 未安装 IK 中文分词插件。")
                            logger.warning("将回退使用标准分词器。对于中文搜索，强烈建议安装 IK 插件以获得更好效果。")
                            
                            # 回退配置：使用标准分词器
                            standard_index_settings = {
                                "mappings": {
                                    "properties": {
                                        "content": {"type": "text"}, # 使用默认的标准分词器
                                        "source": {"type": "keyword"}
                                    }
                                }
                            }
                            # 再次尝试使用标准配置创建
                            self.client.indices.create(index=self.index_name, body=standard_index_settings)
                            logger.info(f"成功使用标准分词器创建索引 '{self.index_name}'。")
                        else:
                            # 如果是其他类型的请求错误，则重新引发异常
                            raise e
            else:
                logger.info(f"索引 '{self.index_name}' 已存在。")
        except Exception as e:
            logger.error(f"创建或检查索引 '{self.index_name}' 时发生严重错误: {e}")

    def index_files(self, files: list):
        """
        高效地索引文件列表。
        根据是否使用 embedding 模式进行不同的处理。
        """
        if not self.client:
            logger.error("Elasticsearch 客户端不可用，无法执行索引。")
            return
            
        logger.info(f"开始处理 {len(files)} 个文件以进行索引...")
        chunks = self._get_chunks(files)
        logger.info(f"从文件中总共提取了 {len(chunks)} 个内容块。")

        if not chunks:
            logger.warning("未能从文件中提取任何内容块，索引过程终止。")
            return

        # 为所有块生成 ID
        for chunk in chunks:
            chunk_content = chunk.get('content', '')
            chunk_source = chunk.get('metadata', {}).get('source', 'unknown')
            sha256 = hashlib.sha256()
            sha256.update(chunk_content.encode('utf-8'))
            sha256.update(chunk_source.encode('utf-8'))
            chunk['id'] = sha256.hexdigest()

        # 打印构建的索引块信息
        for i, chunk in enumerate(chunks):
            logger.info(f"索引块 {i+1}: ID={chunk['id'][:8]}..., 内容长度={len(chunk['content'])}, 源文件={chunk['metadata']['source']}, 标题={chunk['metadata'].get('title', 'N/A')}, chunk_id={chunk['metadata'].get('chunk_id', 'N/A')}, 内容预览='{chunk['content'][:100]}...'")
        
        # 高效地筛选出需要索引的新块
        new_chunks = self._filter_existing_chunks_efficiently(chunks)

        if new_chunks:
            logger.info(f'发现 {len(new_chunks)} 个新的文档块，开始向 Elasticsearch 批量索引...')
            
            if self.use_embedding:
                # 为使用 embedding 的文档块生成向量
                actions = []
                failed_embeddings_count = 0
                for chunk in new_chunks:
                    # 生成 embedding 向量
                    content_embedding = self._get_embedding(chunk['content'])
                    if content_embedding:
                        action = {
                            "_op_type": "index",
                            "_index": self.index_name,
                            "_id": chunk['id'],
                            "_source": {
                                "content": chunk['content'],
                                "source": chunk['metadata']['source'],
                                "token": chunk.get('token', 0),
                                "content_vector": content_embedding
                            }
                        }
                        actions.append(action)
                    else:
                        failed_embeddings_count += 1
                        logger.warning(f"未能为块 {chunk['id']} 生成 embedding 向量，跳过索引。")
                
                if failed_embeddings_count > 0:
                    logger.warning(f"总共 {len(new_chunks)} 个文档块中，有 {failed_embeddings_count} 个因无法生成 embedding 而跳过索引。")
                    logger.warning("请检查 embedding 服务配置后再进行索引。")
            else:
                # 普通 BM25 索引
                actions = [{
                    "_op_type": "index",
                    "_index": self.index_name,
                    "_id": chunk['id'],
                    "_source": {
                        "content": chunk['content'],
                        "source": chunk['metadata']['source'],
                        "token": chunk.get('token', 0)
                    }
                } for chunk in new_chunks]
            
            try:
                successes, errors = helpers.bulk(self.client, actions, refresh=True, raise_on_error=False)
                logger.info(f"成功索引 {successes} 个新文档块。")
                if errors:
                    logger.error(f"批量索引过程中发生 {len(errors)} 个错误。第一个错误详情: {errors[0]}")
            except helpers.BulkIndexError as e:
                logger.error(f"批量索引时发生严重错误: {len(e.errors)} 个文档索引失败。")
        else:
            logger.info("所有文件内容均已在 Elasticsearch 中建立索引，无需更新。")

    def _get_chunks(self, files: list) -> list:
        """从文件列表中提取并返回所有文本块。"""
        all_chunks = []
        for file_path in files:
            try:
                # 1. 准备 JSON 字符串参数
                params_str = json.dumps({'url': file_path})

                # 2. 调用 DocParser，它会返回一个 JSON 字符串
                parsed_content_str = self.parser.call(
                    params=params_str,
                    use_cache=False  # 强制重新解析，忽略缓存
                )

                # 3. 解析返回的 JSON 字符串
                parsed_record = json.loads(parsed_content_str)
                
                # 检查解析后的记录是否出错
                if 'error' in parsed_record:
                    logger.error(f"解析文件 '{file_path}' 时返回错误: {parsed_record['error']}")
                    continue

                # 从记录中提取 'raw' 块
                chunks_data = parsed_record.get('raw', [])
                
                # 为每个块添加源文件信息
                for chunk in chunks_data:
                    if 'metadata' in chunk and 'source' not in chunk['metadata']:
                         chunk['metadata']['source'] = os.path.basename(file_path)
                    all_chunks.append(chunk)

            except Exception as e:
                logger.error(f"处理文件 '{file_path}' 时出错: {e}", exc_info=True)
        return all_chunks

    def _filter_existing_chunks_efficiently(self, chunks: list) -> list:
        """
        使用 mget 高效地从块列表中筛选出尚未在ES中索引的块。
        """
        if not chunks:
            return []

        # 1. 为所有块生成 ID
        for chunk in chunks:
            chunk_content = chunk.get('content', '')
            chunk_source = chunk.get('source', 'unknown')
            sha256 = hashlib.sha256()
            sha256.update(chunk_content.encode('utf-8'))
            sha256.update(chunk_source.encode('utf-8'))
            chunk['id'] = sha256.hexdigest()

        doc_ids = [chunk['id'] for chunk in chunks]
        
        # 2. 使用 mget 一次性检查所有 ID 是否存在
        try:
            response = self.client.mget(index=self.index_name, body={'ids': doc_ids})
            existing_ids = {doc['_id'] for doc in response['docs'] if doc['found']}
            logger.info(f"在 Elasticsearch 中发现 {len(existing_ids)} 个已存在的文档块。")
        except Exception as e:
            logger.error(f"使用 mget 检查文档是否存在时出错: {e}。将假定所有块都是新的。")
            existing_ids = set()

        # 3. 筛选出新块
        new_chunks = [chunk for chunk in chunks if chunk['id'] not in existing_ids]
        logger.info(f"筛选出 {len(new_chunks)} 个新块需要索引。")
        return new_chunks

    def search(self, query: str, max_ref_token: int) -> list:
        """
        在 Elasticsearch 中执行搜索，并根据 max_ref_token 限制返回结果。
        根据是否使用 embedding 模式执行不同的搜索策略。
        """
        if not self.client:
            logger.error("Elasticsearch 客户端不可用，无法执行搜索。")
            return []
        
        logger.info(f"正在使用查询语句在 Elasticsearch 中搜索: '{query}'")
        
        if self.use_embedding:
            # 执行向量搜索 (k-NN)
            logger.info("使用向量搜索 (k-NN) 模式")
            
            # 为查询语句生成向量
            query_vector = self._get_embedding(query)
            if not query_vector:
                logger.error("无法为查询生成向量，将回退到普通搜索模式。")
                # 回退到普通搜索模式
                search_body = {
                    "query": {
                        "match": {
                            "content": query
                        }
                    },
                    "size": 100  
                }
            else:
                # 执行 k-NN 搜索
                search_body = {
                    "knn": {
                        "field": "content_vector",
                        "query_vector": query_vector,
                        "k": 10,  # 获取前10个最相似的结果
                        "num_candidates": 100
                    },
                    "size": 10
                }
        else:
            # 执行 BM25 搜索
            logger.info("使用 BM25 搜索模式")
            
            # 请求一批候选结果（例如 100 个），以便本地筛选
            search_body = {
                "query": {
                    "match": {
                        "content": query
                    }
                },
                "size": 100  
            }
        
        try:
            response = self.client.search(index=self.index_name, body=search_body)
            hits = response['hits']['hits']
            
            # 根据是否使用 embedding 过滤结果
            if self.use_embedding:
                # 过滤低于相似度阈值的结果
                filtered_hits = [hit for hit in hits if hit['_score'] >= self.similarity_threshold]
                logger.info(f"从 {len(hits)} 个向量搜索结果中过滤出 {len(filtered_hits)} 个满足相似度阈值({self.similarity_threshold})的结果")
                hits = filtered_hits
            
            # 根据 max_ref_token 筛选结果
            selected_hits = []
            total_tokens = 0
            for hit in hits:
                token_count = hit['_source'].get('token', 1000) # 从 _source 中获取 token，如不存在则估算一个较大值
                if total_tokens + token_count > max_ref_token:
                    logger.info(f'已达到 max_ref_token ({max_ref_token}) 的上限，停止添加更多结果。')
                    break
                selected_hits.append(hit)
                total_tokens += token_count
            
            logger.info(f"搜索完成，从 {len(hits)} 个候选中筛选出 {len(selected_hits)} 个结果 (总计 an approximately {total_tokens} tokens)。")
            return selected_hits
            
        except Exception as e:
            logger.error(f"Elasticsearch 搜索失败: {e}")
            return []

    def _get_embedding(self, text: str) -> list:
        """使用 Dashscope 的 embedding 模型为文本生成向量。"""
        if not text.strip():
            return []
        
        if not hasattr(self, 'embedding_client') or self.embedding_client is None:
            logger.error("embedding 客户端未初始化，可能是因为缺少 DASHSCOPE_API_KEY 环境变量。请设置环境变量后重试。")
            return []
        
        try:
            response = self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=text,
                dimensions=self.embedding_dimensions,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"获取 embedding 时出错: {e}")
            logger.error("请检查以下配置：")
            logger.error("1. 是否已设置 DASHSCOPE_API_KEY 环境变量")
            logger.error("2. API 密钥是否有效")
            logger.error("3. 网络连接是否正常")
            logger.error("4. 所请求的 embedding 模型是否可用")
            return []