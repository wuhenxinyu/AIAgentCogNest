"""
Elasticsearch 检索工具
"""
import os
import json
from elasticsearch import Elasticsearch, helpers
from .tools.base import BaseTool, register_tool
from pypdf import PdfReader
import warnings

# 忽略不必要的警告
warnings.filterwarnings("ignore", category=UserWarning, module='pypdf')
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from elasticsearch.exceptions import ConnectionError, NotFoundError
except ImportError:
    from urllib3.exceptions import ReadTimeoutError as ConnectionError
    from elasticsearch.exceptions import NotFoundError

class ElasticsearchRetrievalTool(BaseTool):
    """
    一个使用 Elasticsearch 进行文档检索的工具。
    它负责将文件内容分块、索引到 Elasticsearch，并根据用户查询执行搜索。
    """
    name = 'retrieval'  # 关键：工具名保持为 'retrieval'，以覆盖 qwen-agent 的默认内存检索
    description = '从 Elasticsearch 索引的文档中检索与用户查询相关的内容。'
    parameters = [{
        'name': 'query',
        'type': 'string',
        'description': '用户查询的关键词或问题',
        'required': True
    }, {
        'name': 'files',
        'type': 'list',
        'description': '需要检索的文件列表',
        'required': True
    }]

    def __init__(self, cfg: dict = None):
        super().__init__(cfg)
        self.cfg = cfg or {}
        # ES 配置
        self.es_host = self.cfg.get('host', 'http://localhost')
        self.es_port = self.cfg.get('port', 9200)
        self.es_user = self.cfg.get('user', '')
        self.es_password = self.cfg.get('password', '') # 请替换为您的密码
        self.index_name = self.cfg.get('index_name', 'qwen_agent_rag_index')
        self.chunk_size = self.cfg.get('chunk_size', 500)
        self.client = None
        self._connect()
        self._create_index_if_not_exists()

    def _connect(self):
        """建立到 Elasticsearch 的连接"""
        try:
            self.client = Elasticsearch(
                f"{self.es_host}:{self.es_port}",
                basic_auth=(self.es_user, self.es_password),
                verify_certs=False
            )
            if not self.client.ping():
                raise ConnectionError("连接失败，请检查 Elasticsearch 服务状态和配置。")
            print("成功连接到 Elasticsearch！")
        except ConnectionError as e:
            print(f"无法连接到 Elasticsearch: {e}")
            self.client = None

    def _create_index_if_not_exists(self):
        """如果索引不存在，则创建它并定义映射"""
        if self.client and not self.client.indices.exists(index=self.index_name):
            print(f"索引 '{self.index_name}' 不存在，正在创建...")
            mapping = {
                "properties": {
                    "file_name": {"type": "keyword"},
                    "chunk_id": {"type": "integer"},
                    "content": {
                        "type": "text",
                        "analyzer": "ik_max_word",  # 使用 IK 分词器（如果已安装）
                        "search_analyzer": "ik_smart"
                    }
                }
            }
            try:
                self.client.indices.create(index=self.index_name, mappings=mapping)
            except Exception:
                # 如果没有 IK 分词器，则使用标准分词器
                mapping['properties']['content'].pop('analyzer')
                mapping['properties']['content'].pop('search_analyzer')
                self.client.indices.create(index=self.index_name, mappings=mapping)
            print(f"索引 '{self.index_name}' 创建成功。")

    def call(self, params: str, **kwargs) -> str:
        """工具调用的主入口"""
        if not self.client:
            return json.dumps([{'error': 'Elasticsearch 未连接，无法执行检索。'}], ensure_ascii=False)

        try:
            params = json.loads(params)
            query = params.get('query', '')
            files = params.get('files', [])
        except json.JSONDecodeError:
            return json.dumps([{'error': '输入参数格式错误，必须是有效的 JSON 字符串。'}], ensure_ascii=False)

        # 步骤 1: 索引新文件或更新文件
        self._index_files(files)

        # 步骤 2: 执行搜索
        if not query:
            return json.dumps([], ensure_ascii=False)
        
        try:
            search_body = {
                "query": {
                    "match": {
                        "content": query
                    }
                },
                "size": 5 # 返回最相关的5个块
            }
            response = self.client.search(index=self.index_name, body=search_body)

            # 步骤 3: 格式化并返回结果
            results = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                results.append({
                    "source": f"{source['file_name']} (块 {source['chunk_id']})",
                    "content": source['content'],
                    "score": hit['_score']
                })
            
            # 按得分降序排序
            results.sort(key=lambda x: x['score'], reverse=True)
            return json.dumps(results, ensure_ascii=False)

        except Exception as e:
            return json.dumps([{'error': f'搜索时发生错误: {str(e)}'}], ensure_ascii=False)

    def _index_files(self, files: list):
        """处理文件，将其分块并索引到 ES"""
        for file_path in files:
            if not os.path.exists(file_path):
                print(f"警告: 文件不存在，跳过索引: {file_path}")
                continue

            # 简单检查：如果文件已存在，则跳过（可优化为检查文件修改时间或哈希值）
            if self._is_file_indexed(file_path):
                print(f"文件 '{os.path.basename(file_path)}' 已被索引，跳过。")
                continue
            
            print(f"正在索引新文件: {file_path}...")
            try:
                content = self._read_file_content(file_path)
                chunks = self._chunk_text(content)
                self._bulk_index_chunks(file_path, chunks)
                print(f"文件 '{os.path.basename(file_path)}' 索引完成。")
            except Exception as e:
                print(f"处理文件 '{file_path}' 时出错: {e}")

    def _is_file_indexed(self, file_path: str) -> bool:
        """检查特定文件是否已被索引"""
        file_name = os.path.basename(file_path)
        query = {"query": {"term": {"file_name": file_name}}}
        try:
            response = self.client.count(index=self.index_name, body=query)
            return response['count'] > 0
        except NotFoundError:
            return False # 索引可能还不存在
        except Exception:
            return False # 其他错误，保守地认为未索引

    def _read_file_content(self, file_path: str) -> str:
        """根据文件类型读取文件内容"""
        content = ""
        if file_path.lower().endswith('.pdf'):
            reader = PdfReader(file_path)
            for page in reader.pages:
                content += page.extract_text() or ""
        elif file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        else:
            raise ValueError(f"不支持的文件类型: {file_path}")
        return content

    def _chunk_text(self, text: str) -> list:
        """将长文本切分成指定大小的块"""
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    def _bulk_index_chunks(self, file_path: str, chunks: list):
        """使用 bulk API 高效地批量索引块"""
        file_name = os.path.basename(file_path)
        actions = []
        for i, chunk in enumerate(chunks):
            action = {
                "_index": self.index_name,
                "_source": {
                    "file_name": file_name,
                    "chunk_id": i + 1,
                    "content": chunk,
                }
            }
            actions.append(action)
        
        if actions:
            helpers.bulk(self.client, actions)
            self.client.indices.refresh(index=self.index_name) 