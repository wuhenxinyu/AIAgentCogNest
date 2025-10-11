# qwen_agent/tools/es_retrieval.py
import json
from typing import Dict, List
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.tools.doc_parser import DocParser
from qwen_agent.utils.utils import print_traceback
from qwen_agent.searcher.elasticsearch_searcher import ElasticsearchSearcher
from qwen_agent.settings import DEFAULT_MAX_REF_TOKEN

class ESRetrievalTool(BaseTool):
    """
    一个使用 Elasticsearch 作为后端的检索工具。
    它协调文档解析、分块、索引和搜索的整个流程。
    """
    name = 'retrieval'  # 保持名称为 'retrieval' 以便在 Memory 中进行替换
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
        self.max_ref_token = self.cfg.get('max_ref_token', DEFAULT_MAX_REF_TOKEN)
        # 初始化 ES 搜索器，它内部会管理文档解析
        self.searcher = ElasticsearchSearcher(cfg=self.cfg)

    def call(self, params: dict, **kwargs) -> str:
        """
        工具调用的主入口。
        1. (如有需要) 将所有文件传递给 searcher 进行高效索引。
        2. 执行搜索。
        3. 返回格式化的结果。
        """
        query_input = params.get('query', '')
        files = params.get('files', [])

        # 1. 高效地索引所有文件
        # ElasticsearchSearcher 内部会处理分块、去重和批量索引
        if files and isinstance(files, list):
            self.searcher.index_files(files)
        
        # 2. 如果没有查询，直接返回空列表
        if not query_input:
            return json.dumps([], ensure_ascii=False)
        
        # 3. 解析来自 Memory 模块的复杂 JSON 查询
        try:
            # 尝试将输入解析为 JSON 对象
            query_obj = json.loads(query_input)
            if isinstance(query_obj, dict):
                # 优先使用 'text' 字段作为查询
                query = query_obj.get('text', '')
                # 如果 'text' 为空，尝试使用中文关键词列表
                if not query and 'keywords_zh' in query_obj and query_obj['keywords_zh']:
                    query = ' '.join(query_obj['keywords_zh'])
                # 如果还是空，则退回使用原始输入
                if not query:
                    query = query_input
            else:
                # 如果是 JSON 但不是字典（例如列表），则按原样使用
                query = query_input
        except (json.JSONDecodeError, TypeError):
            # 如果输入不是有效的 JSON，则按原样使用它
            query = query_input
            
        # 4. 执行搜索
        search_results = self.searcher.search(query, max_ref_token=self.max_ref_token)
        
        # 5. 格式化并返回结果
        # 将结果格式化为 Agent 期望的 {'url': ..., 'text': ...} 格式
        formatted_results = [
            {
                'url': hit.get('_source', {}).get('source', 'N/A'),
                'text': hit.get('_source', {}).get('content', '')
            } for hit in search_results
        ]
        
        return json.dumps(formatted_results, ensure_ascii=False)

    def _format_error(self, error_message: str) -> str:
        return json.dumps([{'error': error_message}], ensure_ascii=False) 