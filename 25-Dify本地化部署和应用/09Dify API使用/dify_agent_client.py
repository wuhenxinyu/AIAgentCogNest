"""
Dify Agent 客户端
用于调用Dify平台的Agent API
支持聊天、完成和工作流应用类型
"""

import requests
import json
import time
from typing import Dict, Any, Optional


class DifyAgentClient:
    """Dify Agent API 客户端类"""
    
    def __init__(self, base_url: str, api_key: str):
        """
        初始化Dify Agent客户端
        
        Args:
            base_url (str): Dify API的基础URL
            api_key (str): 应用的API密钥
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
    
    def chat_completion(self, 
                       user_input: str, 
                       user_id: str = "default_user",
                       conversation_id: Optional[str] = None,
                       stream: bool = False,
                       app_type: str = "auto") -> Dict[str, Any]:
        """
        调用Dify Agent进行对话
        
        Args:
            user_input (str): 用户输入的消息
            user_id (str): 用户ID，默认为"default_user"
            conversation_id (str, optional): 会话ID，用于维持对话上下文
            stream (bool): 是否使用流式响应，默认为False
            app_type (str): 应用类型，"chat"、"completion"、"workflow"或"auto"（自动检测）
            
        Returns:
            Dict[str, Any]: API响应结果
        """
        if app_type == "auto":
            # 先尝试chat端点，如果失败则尝试completion端点，最后尝试workflow端点
            result = self._try_chat_endpoint(user_input, user_id, conversation_id, stream)
            if result.get("error") and "not_chat_app" in str(result.get("message", "")):
                print("检测到非聊天应用，切换到completion端点...")
                result = self._try_completion_endpoint(user_input, user_id, stream)
                if result.get("error") and "app_unavailable" in str(result.get("message", "")):
                    print("检测到非完成应用，切换到工作流端点...")
                    return self._try_workflow_endpoint(user_input, user_id, stream)
            return result
        elif app_type == "chat":
            return self._try_chat_endpoint(user_input, user_id, conversation_id, stream)
        elif app_type == "completion":
            return self._try_completion_endpoint(user_input, user_id, stream)
        elif app_type == "workflow":
            return self._try_workflow_endpoint(user_input, user_id, stream)
        else:
            return {
                "error": True,
                "message": f"不支持的应用类型: {app_type}"
            }
    
    def _try_chat_endpoint(self, user_input: str, user_id: str, conversation_id: Optional[str], stream: bool) -> Dict[str, Any]:
        """尝试使用chat端点"""
        url = f"{self.base_url}/chat-messages"
        
        # 构建基础payload
        payload = {
            "inputs": {},
            "query": user_input,
            "response_mode": "streaming" if stream else "blocking",
            "user": user_id
        }
        
        # 只有当conversation_id不为None且不为空字符串时才添加
        if conversation_id:
            payload["conversation_id"] = conversation_id
        
        try:
            if stream:
                return self._handle_streaming_response(url, payload)
            else:
                return self._handle_blocking_response(url, payload)
                
        except requests.exceptions.RequestException as e:
            return {
                "error": True,
                "message": f"请求失败: {str(e)}"
            }
        except Exception as e:
            return {
                "error": True,
                "message": f"未知错误: {str(e)}"
            }
    
    def _try_completion_endpoint(self, user_input: str, user_id: str, stream: bool) -> Dict[str, Any]:
        """尝试使用completion端点"""
        url = f"{self.base_url}/completion-messages"
        
        # 根据官方文档，completion端点的正确格式
        payload = {
            "inputs": {},  # 空对象，不是包含query的对象
            "response_mode": "streaming" if stream else "blocking",
            "user": user_id
        }
        
        try:
            if stream:
                return self._handle_streaming_response(url, payload)
            else:
                return self._handle_blocking_response(url, payload)
                
        except requests.exceptions.RequestException as e:
            return {
                "error": True,
                "message": f"请求失败: {str(e)}"
            }
        except Exception as e:
            return {
                "error": True,
                "message": f"未知错误: {str(e)}"
            }
    
    def _try_workflow_endpoint(self, user_input: str, user_id: str, stream: bool) -> Dict[str, Any]:
        """尝试使用workflow端点"""
        url = f"{self.base_url}/workflows/run"
        
        # 工作流端点的payload格式
        payload = {
            "inputs": {"query": user_input},  # 工作流通常需要在inputs中传递参数
            "response_mode": "streaming" if stream else "blocking",
            "user": user_id
        }
        
        try:
            if stream:
                return self._handle_workflow_streaming_response(url, payload)
            else:
                return self._handle_workflow_blocking_response(url, payload)
                
        except requests.exceptions.RequestException as e:
            return {
                "error": True,
                "message": f"请求失败: {str(e)}"
            }
        except Exception as e:
            return {
                "error": True,
                "message": f"未知错误: {str(e)}"
            }
    
    def run_workflow(self, 
                    inputs: Dict[str, Any], 
                    user_id: str = "default_user",
                    stream: bool = False) -> Dict[str, Any]:
        """
        直接运行工作流
        
        Args:
            inputs (dict): 工作流输入参数
            user_id (str): 用户ID，默认为"default_user"
            stream (bool): 是否使用流式响应，默认为False
            
        Returns:
            Dict[str, Any]: API响应结果
        """
        url = f"{self.base_url}/workflows/run"
        
        payload = {
            "inputs": inputs,
            "response_mode": "streaming" if stream else "blocking",
            "user": user_id
        }
        
        try:
            if stream:
                return self._handle_workflow_streaming_response(url, payload)
            else:
                return self._handle_workflow_blocking_response(url, payload)
                
        except requests.exceptions.RequestException as e:
            return {
                "error": True,
                "message": f"请求失败: {str(e)}"
            }
        except Exception as e:
            return {
                "error": True,
                "message": f"未知错误: {str(e)}"
            }
    
    def completion_message(self, 
                          user_input: str, 
                          user_id: str = "default_user",
                          stream: bool = False,
                          inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        直接调用completion端点，支持自定义inputs
        
        Args:
            user_input (str): 用户输入的消息
            user_id (str): 用户ID，默认为"default_user"
            stream (bool): 是否使用流式响应，默认为False
            inputs (dict, optional): 自定义输入参数
            
        Returns:
            Dict[str, Any]: API响应结果
        """
        url = f"{self.base_url}/completion-messages"
        
        # 如果没有提供inputs，尝试不同的格式
        if inputs is None:
            # 尝试多种可能的inputs格式
            possible_inputs = [
                {},  # 空对象（官方文档格式）
                {"text": user_input},  # 文本格式
                {"query": user_input},  # 查询格式
                {"input": user_input},  # 输入格式
                {"prompt": user_input}  # 提示格式
            ]
        else:
            possible_inputs = [inputs]
        
        for input_format in possible_inputs:
            payload = {
                "inputs": input_format,
                "response_mode": "streaming" if stream else "blocking",
                "user": user_id
            }
            
            try:
                if stream:
                    result = self._handle_streaming_response(url, payload)
                else:
                    result = self._handle_blocking_response(url, payload)
                
                # 如果成功，返回结果
                if not result.get("error"):
                    return result
                
                # 如果是app_unavailable错误，尝试下一种格式
                if "app_unavailable" in str(result.get("message", "")):
                    continue
                else:
                    # 其他错误直接返回
                    return result
                    
            except Exception as e:
                continue
        
        # 所有格式都失败了
        return {
            "error": True,
            "message": "所有输入格式都失败了，请检查应用配置和API密钥"
        }
    
    def _handle_blocking_response(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """处理阻塞式响应"""
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            
            # 打印调试信息
            print(f"请求URL: {url}")
            print(f"请求头: {self.headers}")
            print(f"请求体: {json.dumps(payload, ensure_ascii=False, indent=2)}")
            print(f"响应状态码: {response.status_code}")
            
            if response.status_code != 200:
                print(f"响应内容: {response.text}")
                return {
                    "error": True,
                    "message": f"HTTP {response.status_code}: {response.text}"
                }
            
            result = response.json()
            return {
                "error": False,
                "data": result,
                "answer": result.get("answer", ""),
                "conversation_id": result.get("conversation_id", ""),
                "message_id": result.get("message_id", "")
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "error": True,
                "message": f"网络请求异常: {str(e)}"
            }
        except json.JSONDecodeError as e:
            return {
                "error": True,
                "message": f"JSON解析错误: {str(e)}"
            }
    
    def _handle_workflow_blocking_response(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """处理工作流阻塞式响应"""
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            
            # 打印调试信息
            print(f"请求URL: {url}")
            print(f"请求头: {self.headers}")
            print(f"请求体: {json.dumps(payload, ensure_ascii=False, indent=2)}")
            print(f"响应状态码: {response.status_code}")
            
            if response.status_code != 200:
                print(f"响应内容: {response.text}")
                return {
                    "error": True,
                    "message": f"HTTP {response.status_code}: {response.text}"
                }
            
            result = response.json()
            
            # 工作流响应格式可能不同
            outputs = result.get("data", {}).get("outputs", {})
            
            # 尝试从outputs中提取答案
            answer = ""
            if isinstance(outputs, dict):
                # 常见的输出字段名
                for key in ["answer", "result", "output", "text", "response"]:
                    if key in outputs:
                        answer = str(outputs[key])
                        break
                
                # 如果没有找到，使用第一个值
                if not answer and outputs:
                    answer = str(list(outputs.values())[0])
            
            return {
                "error": False,
                "data": result,
                "answer": answer,
                "outputs": outputs,
                "workflow_run_id": result.get("workflow_run_id", ""),
                "task_id": result.get("task_id", "")
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "error": True,
                "message": f"网络请求异常: {str(e)}"
            }
        except json.JSONDecodeError as e:
            return {
                "error": True,
                "message": f"JSON解析错误: {str(e)}"
            }
    
    def _handle_streaming_response(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """处理流式响应"""
        try:
            response = requests.post(url, headers=self.headers, json=payload, stream=True, timeout=60)
            
            # 打印调试信息
            print(f"请求URL: {url}")
            print(f"请求头: {self.headers}")
            print(f"请求体: {json.dumps(payload, ensure_ascii=False, indent=2)}")
            print(f"响应状态码: {response.status_code}")
            
            if response.status_code != 200:
                print(f"响应内容: {response.text}")
                return {
                    "error": True,
                    "message": f"HTTP {response.status_code}: {response.text}"
                }
            
            full_answer = ""
            conversation_id = ""
            message_id = ""
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])
                            if data.get('event') == 'message':
                                full_answer += data.get('answer', '')
                            elif data.get('event') == 'message_end':
                                conversation_id = data.get('conversation_id', '')
                                message_id = data.get('id', '')
                        except json.JSONDecodeError:
                            continue
            
            return {
                "error": False,
                "answer": full_answer,
                "conversation_id": conversation_id,
                "message_id": message_id
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "error": True,
                "message": f"网络请求异常: {str(e)}"
            }
    
    def _handle_workflow_streaming_response(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """处理工作流流式响应"""
        try:
            response = requests.post(url, headers=self.headers, json=payload, stream=True, timeout=60)
            
            # 打印调试信息
            print(f"请求URL: {url}")
            print(f"请求头: {self.headers}")
            print(f"请求体: {json.dumps(payload, ensure_ascii=False, indent=2)}")
            print(f"响应状态码: {response.status_code}")
            
            if response.status_code != 200:
                print(f"响应内容: {response.text}")
                return {
                    "error": True,
                    "message": f"HTTP {response.status_code}: {response.text}"
                }
            
            full_answer = ""
            workflow_run_id = ""
            task_id = ""
            final_outputs = {}
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])
                            event = data.get('event', '')
                            
                            if event == 'workflow_started':
                                workflow_run_id = data.get('workflow_run_id', '')
                                task_id = data.get('task_id', '')
                            elif event == 'workflow_finished':
                                final_outputs = data.get('data', {}).get('outputs', {})
                                # 尝试从outputs中提取答案
                                if isinstance(final_outputs, dict):
                                    for key in ["answer", "result", "output", "text", "response"]:
                                        if key in final_outputs:
                                            full_answer = str(final_outputs[key])
                                            break
                                    if not full_answer and final_outputs:
                                        full_answer = str(list(final_outputs.values())[0])
                            elif event == 'node_finished':
                                # 节点完成，可能包含中间结果
                                node_outputs = data.get('data', {}).get('outputs', {})
                                if isinstance(node_outputs, dict):
                                    for key in ["answer", "result", "output", "text", "response"]:
                                        if key in node_outputs:
                                            full_answer += str(node_outputs[key])
                                            break
                        except json.JSONDecodeError:
                            continue
            
            return {
                "error": False,
                "answer": full_answer,
                "outputs": final_outputs,
                "workflow_run_id": workflow_run_id,
                "task_id": task_id
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "error": True,
                "message": f"网络请求异常: {str(e)}"
            }
    
    def get_conversation_messages(self, conversation_id: str, user_id: str = "default_user") -> Dict[str, Any]:
        """
        获取会话历史消息
        
        Args:
            conversation_id (str): 会话ID
            user_id (str): 用户ID
            
        Returns:
            Dict[str, Any]: 会话消息列表
        """
        url = f"{self.base_url}/messages"
        params = {
            "conversation_id": conversation_id,
            "user": user_id
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return {
                "error": False,
                "data": result
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "error": True,
                "message": f"获取会话消息失败: {str(e)}"
            }