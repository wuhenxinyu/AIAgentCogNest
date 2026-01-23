import os
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, List, Union, Literal
from pydantic import BaseModel, Field
from openai import OpenAI
from src.api_requests import BaseOpenaiProcessor, AsyncOpenaiProcessor
import tiktoken
from tqdm import tqdm
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import time

message_queue = Queue()

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            message_queue.put((record.levelno, msg))
        except Exception:
            self.handleError(record)

def process_messages():
    while not message_queue.empty():
        level, msg = message_queue.get_nowait()
        tqdm.write(msg)

# TableSerializer：表格序列化主流程类，支持同步/异步LLM表格结构化
class TableSerializer(BaseOpenaiProcessor):
    def __init__(self, preserve_temp_files: bool = True):
        super().__init__()
        self.preserve_temp_files = preserve_temp_files
        os.makedirs('./temp', exist_ok=True)
        
        self.logger = logging.getLogger('TableSerializer')
        self.logger.setLevel(logging.INFO)
        
        self.logger.handlers.clear()
        
        handler = TqdmLoggingHandler()
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        self.logger.addHandler(handler)
        
        self.logger.propagate = False

    def _get_table_context(self, json_report, target_table_index):
        # 获取表格所在页的上下文文本（前后各最多3个块）
        table_info = next(table for table in json_report["tables"] if table["table_id"] == target_table_index)
        page_num = table_info["page"]
        
        page_content = next(
            (page["content"] for page in json_report["content"] if page["page"] == page_num),
            []
        )
        
        if not page_content:
            self.logger.warning(f"Page {page_num} not found for table {target_table_index}")
            return "", ""

        # 定位目标表格在页面中的位置
        current_table_position = -1
        for i, block in enumerate(page_content):
            if block["type"] == "table" and block.get("table_id") == target_table_index:
                current_table_position = i
                break

        # 查找前一个表格位置
        previous_table_position = -1
        for i in range(current_table_position-1, -1, -1):
            if page_content[i]["type"] == "table":
                previous_table_position = i
                break

        # 查找下一个表格位置
        next_table_position = -1
        for i in range(current_table_position + 1, len(page_content)):
            if page_content[i]["type"] == "table":
                next_table_position = i
                break

        # 获取当前表格上方的块
        start_position = previous_table_position + 1 if previous_table_position != -1 else 0
        context_before = page_content[start_position:current_table_position]

        # 获取当前表格下方的块
        context_after = []
        if next_table_position == -1:
            # 没有下一个表格，取后3个块
            context_after = page_content[current_table_position + 1:current_table_position + 4]
        else:
            # 有下一个表格，取到下一个表格前最多3个块
            blocks_between = next_table_position - (current_table_position + 1)
            if blocks_between > 3:
                context_after = page_content[current_table_position + 1:current_table_position + 4]
            elif blocks_between > 1:
                context_after = page_content[current_table_position + 1:current_table_position + blocks_between]

        context_before = "\n".join(block.get("text", "") for block in context_before if "text" in block)
        context_after = "\n".join(block.get("text", "") for block in context_after if "text" in block)

        return context_before, context_after

    def _send_serialization_request(self, table, context_before, context_after):
        # 构造LLM表格序列化请求，拼接上下文和表格HTML
        user_prompt = ""
        
        if context_before:
            user_prompt += f'Here is additional text before the table that might be relevant (or not):\n"""{context_before}"""\n\n'
        
        user_prompt += f'Here is a table in HTML format:\n"""{table}"""'
        
        if context_after:
            user_prompt += f'\n\nHere is additional text after the table that might be relevant (or not):\n"""{context_after}"""'
        
        system_prompt = TableSerialization.system_prompt
        reponse_schema = TableSerialization.TableBlocksCollection

        answer_dict = self.send_message(
            model='gpt-4o-mini-2024-07-18',
            temperature=0,
            system_content=system_prompt,
            human_content=user_prompt,
            is_structured=True,
            response_format=reponse_schema
        )

        input_message = user_prompt + system_prompt + str(reponse_schema.schema())
        input_tokens = self.count_tokens(input_message)
        output_tokens = self.count_tokens(str(answer_dict))

        result = answer_dict
        return result
    
    def _serialize_table(self, json_report: dict, target_table_index: int) -> dict:
        # 序列化单个表格，获取上下文并调用LLM
        context_before, context_after = self._get_table_context(json_report, target_table_index)
        table_info = next(table for table in json_report["tables"] if table["table_id"] == target_table_index)
        table_content = table_info["html"]
        result = self._send_serialization_request(
            table=table_content,
            context_before=context_before,
            context_after=context_after
        )
        return result

    def serialize_tables(self, json_report: dict) -> dict:
        """批量处理报告中所有表格，序列化结果写入table['serialized']"""
        for table in json_report["tables"]:
            table_index = table["table_id"]
            # 获取当前表格的序列化结果
            serialization_result = self._serialize_table(
                json_report=json_report,
                target_table_index=table_index
            )
            # 写入序列化结果
            table["serialized"] = serialization_result
        return json_report

    async def async_serialize_tables(
        self, 
        json_report: dict,
        requests_filepath: str = './temp_async_llm_requests.jsonl',
        results_filepath: str = './temp_async_llm_results.jsonl'
    ) -> dict:
        """异步批量处理报告中所有表格，适合大规模并发"""
        queries = []
        table_indices = []
        
        for table in json_report["tables"]:
            table_index = table["table_id"]
            table_indices.append(table_index)
            
            context_before, context_after = self._get_table_context(json_report, table_index)
            table_info = next(table for table in json_report["tables"] if table["table_id"] == table_index)
            table_content = table_info["html"]
            
            # 构造异步请求query
            query = ""
            if context_before:
                query += f'Here is additional text before the table that might be relevant (or not):\n"""{context_before}"""\n\n'
            query += f'Here is a table in HTML format:\n"""{table_content}"""'
            if context_after:
                query += f'\n\nHere is additional text after the table that might be relevant (or not):\n"""{context_after}"""'
            
            queries.append(query)

        results = await AsyncOpenaiProcessor().process_structured_ouputs_requests(
            model='gpt-4o-mini-2024-07-18',
            temperature=0,
            system_content=TableSerialization.system_prompt,
            queries=queries,
            response_format=TableSerialization.TableBlocksCollection,
            preserve_requests=False,
            preserve_results=False,
            logging_level=20,
            requests_filepath=requests_filepath,
            save_filepath=results_filepath,
        )

        # Add results back to json_report
        for table_index, result in zip(table_indices, results):
            table_info = next(table for table in json_report["tables"] if table["table_id"] == table_index)
            
            new_table = {}
            for key, value in table_info.items():
                new_table[key] = value
                if key == "html":
                    new_table["serialized"] = result["answer"]
            
            for i, table in enumerate(json_report["tables"]):
                if table["table_id"] == table_index:
                    json_report["tables"][i] = new_table

        return json_report

    def process_file(self, json_path: Path) -> None:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_report = json.load(f)
            
            thread_id = threading.get_ident()
            requests_filepath = f'./temp/async_llm_requests_{thread_id}.jsonl'
            results_filepath = f'./temp/async_llm_results_{thread_id}.jsonl'
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                updated_report = loop.run_until_complete(self.async_serialize_tables(
                    json_report,
                    requests_filepath=requests_filepath,
                    results_filepath=results_filepath
                ))
            finally:
                loop.close()
                try:
                    os.remove(requests_filepath)
                    os.remove(results_filepath)
                except FileNotFoundError:
                    pass
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(updated_report, f, indent=2, ensure_ascii=False)
                
        except json.JSONDecodeError as e:
            self.logger.error("JSON Error in %s: %s", json_path.name, str(e))
            raise
        except Exception as e:
            self.logger.error("Error processing %s: %s", json_path.name, str(e))
            raise

    def process_directory_parallel(self, input_dir: Path, max_workers: int = 5):
        """Process JSON files in parallel using thread pool.
        
        Args:
            input_dir: Path to directory containing JSON files
            max_workers: Maximum number of threads to use
        """
        self.logger.info("Starting parallel table serialization...")
        
        json_files = list(input_dir.glob("*.json"))
        
        if not json_files:
            self.logger.warning("No JSON files found in %s", input_dir)
            return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(
                total=len(json_files),
                desc="Processing files",
                mininterval=1.0,
                maxinterval=5.0,
                smoothing=0.3
            ) as pbar:
                futures = []
                for json_file in json_files:
                    future = executor.submit(self.process_file, json_file)
                    future.add_done_callback(lambda p: pbar.update(1))
                    futures.append(future)
                
                while futures:
                    process_messages()
                    
                    done_futures = []
                    for future in futures:
                        if future.done():
                            done_futures.append(future)
                            try:
                                future.result()
                            except Exception as e:
                                self.logger.error(str(e))
                    
                    for future in done_futures:
                        futures.remove(future)
                    
                    time.sleep(0.1)

        process_messages()
        self.logger.info("Table serialization completed!")


class TableSerialization:
        
    system_prompt = (
        "You are a table serialization agent.\n"
        "Your task is to create a set of contextually independent blocks of information based on the provided table and surrounding text.\n"
        "These blocks must be totally context-independent because they will be used as separate chunk to populate database."
    )

    class SerializedInformationBlock(BaseModel):
        "A single self-contained information block enriched with comprehensive context"

        subject_core_entity: str = Field(description="A primary focus of what this block is about. Usually located in a row header. If one row in the table doesn't make sense without neighboring rows, you can merge information from neighboring rows into one block")
        information_block: str = Field(description=(
    "Detailed information about the chosen core subject from tables and additional texts. Information SHOULD include:\n"
    "1. All related header information\n"
    "2. All related units and their descriptions\n"
    "    2.1. If header is Total, always write additional context about what this total represents in this block!\n"
    "3. All additional info for context enrichment to make ensure complete context-independency if it present in whole table. This can include:\n"
    "    - The name of the table\n"
    "    - Additional footnotes\n"
    "    - The currency used\n"
    "    - The way amounts are presented\n"
    "    - Anything else that can make context even slightly richer\n"
    "SKIPPING ANY VALUABLE INFORMATION WILL BE HEAVILY PENALIZED!"
    ))

    class TableBlocksCollection(BaseModel):
        """Collection of serialized table blocks with their core entities and header relationships"""

        subject_core_entities_list: List[str] = Field(
            description="A complete list of core entities. Keep in mind, empty headers are possible - they should also be interpreted and listed (Usually it's a total or something similar). In most cases each row header represents a core entity")
        relevant_headers_list: List[str] = Field(description="A list of ALL headers relevant to the subject. These headers will serve as keys in each information block. In most cases each column header represents a core entity")
        information_blocks: List["TableSerialization.SerializedInformationBlock"] = Field(description="Complete list of fully described context-independent information blocks")
