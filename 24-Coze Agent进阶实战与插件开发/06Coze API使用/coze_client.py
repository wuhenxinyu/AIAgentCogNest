"""
基于 cozepy 的 Coze API 客户端
用于与Coze智能体进行交互
"""
import os
from typing import List, Optional, Generator, Dict, Any
from cozepy import (
    Coze, 
    TokenAuth, 
    Message, 
    ChatEventType, 
    MessageContentType,
    ChatEvent,
    ChatStatus
)
from config import COZE_API_TOKEN, COZE_BOT_ID, COZE_CN_BASE_URL, DEFAULT_USER_ID


class CozeClient:
    """基于 cozepy 的 Coze API 客户端类"""
    
    def __init__(self, api_token: str = None, bot_id: str = None, base_url: str = None):
        """
        初始化Coze客户端
        
        Args:
            api_token: Coze API token
            bot_id: 智能体ID
            base_url: API基础URL
        """
        self.api_token = api_token or COZE_API_TOKEN
        self.bot_id = bot_id or COZE_BOT_ID
        self.base_url = base_url or COZE_CN_BASE_URL
        
        # 初始化 Coze 客户端
        self.coze = Coze(
            auth=TokenAuth(token=self.api_token),
            base_url=self.base_url
        )
        
        print(f"✅ Coze客户端初始化成功")
        print(f"📍 API地址: {self.base_url}")
        print(f"🤖 智能体ID: {self.bot_id}")
    
    def chat_stream(self, message: str, user_id: str = None) -> Generator[str, None, None]:
        """
        流式聊天，实时返回智能体的回复
        
        Args:
            message: 用户消息
            user_id: 用户ID
            
        Yields:
            智能体回复的文本片段
        """
        user_id = user_id or DEFAULT_USER_ID
        
        try:
            # 创建流式聊天
            for event in self.coze.chat.stream(
                bot_id=self.bot_id,
                user_id=user_id,
                additional_messages=[Message.build_user_question_text(message)],
            ):
                # 处理消息增量事件
                if event.event == ChatEventType.CONVERSATION_MESSAGE_DELTA:
                    # 检查消息内容是否存在且为文本类型
                    if (hasattr(event.message, 'content') and 
                        event.message.content and
                        hasattr(event.message.content, 'type') and
                        event.message.content.type == MessageContentType.TEXT):
                        yield event.message.content.text
                    elif hasattr(event.message, 'content') and isinstance(event.message.content, str):
                        # 如果content直接是字符串
                        yield event.message.content
                    
        except Exception as e:
            print(f"❌ 流式聊天发生错误: {e}")
            yield f"错误: {str(e)}"
    
    def chat(self, message: str, user_id: str = None) -> Optional[str]:
        """
        普通聊天，返回完整的智能体回复
        
        Args:
            message: 用户消息
            user_id: 用户ID
            
        Returns:
            智能体的完整回复
        """
        user_id = user_id or DEFAULT_USER_ID
        
        try:
            # 使用create_and_poll方法，这是SDK提供的简化方法
            chat_poll = self.coze.chat.create_and_poll(
                bot_id=self.bot_id,
                user_id=user_id,
                additional_messages=[Message.build_user_question_text(message)],
            )
            
            # 检查聊天状态
            if chat_poll.chat.status == ChatStatus.COMPLETED:
                # 从消息列表中提取助手的回复
                for msg in chat_poll.messages:
                    if msg.role == "assistant" and msg.content:
                        return msg.content
                
                return "智能体没有回复内容"
            else:
                return f"聊天未完成，状态: {chat_poll.chat.status}"
            
        except Exception as e:
            print(f"❌ 聊天发生错误: {e}")
            return None
    
    def chat_with_history(self, messages: List[Dict[str, str]], user_id: str = None) -> Optional[str]:
        """
        带历史记录的聊天
        
        Args:
            messages: 消息历史，格式为 [{"role": "user", "content": "..."}, ...]
            user_id: 用户ID
            
        Returns:
            智能体的回复
        """
        user_id = user_id or DEFAULT_USER_ID
        
        try:
            # 构建消息列表
            coze_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    coze_messages.append(Message.build_user_question_text(msg["content"]))
                elif msg["role"] == "assistant":
                    coze_messages.append(Message.build_assistant_answer(msg["content"]))
            
            # 使用create_and_poll方法
            chat_poll = self.coze.chat.create_and_poll(
                bot_id=self.bot_id,
                user_id=user_id,
                additional_messages=coze_messages,
            )
            
            # 检查聊天状态
            if chat_poll.chat.status == ChatStatus.COMPLETED:
                # 从消息列表中提取助手的回复
                for msg in chat_poll.messages:
                    if msg.role == "assistant" and msg.content:
                        return msg.content
                
                return "智能体没有回复内容"
            else:
                return f"聊天未完成，状态: {chat_poll.chat.status}"
            
        except Exception as e:
            print(f"❌ 带历史记录的聊天发生错误: {e}")
            return None
    
    def get_bot_info(self) -> Optional[Dict[str, Any]]:
        """
        获取智能体信息
        
        Returns:
            智能体信息字典
        """
        try:
            bot_info = self.coze.bots.retrieve(bot_id=self.bot_id)
            return {
                "bot_id": bot_info.bot_id,
                "name": bot_info.name,
                "description": bot_info.description,
                "create_time": bot_info.create_time,
                "update_time": bot_info.update_time,
            }
        except Exception as e:
            print(f"❌ 获取智能体信息失败: {e}")
            return None


def interactive_chat():
    """交互式聊天函数"""
    print("🚀 Coze智能体交互式聊天启动！")
    print("💡 输入 'quit' 或 'exit' 退出程序")
    print("💡 输入 'stream' 切换到流式模式")
    print("💡 输入 'normal' 切换到普通模式")
    print("💡 输入 'info' 查看智能体信息")
    print("-" * 60)
    
    client = CozeClient()
    stream_mode = False
    
    # 显示智能体信息
    bot_info = client.get_bot_info()
    if bot_info:
        print(f"🤖 智能体名称: {bot_info.get('name', '未知')}")
        print(f"📝 智能体描述: {bot_info.get('description', '无描述')}")
        print("-" * 60)
    
    while True:
        try:
            user_input = input(f"\n{'[流式]' if stream_mode else '[普通]'} 请输入您的问题: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("👋 再见！")
                break
            
            if user_input.lower() == 'stream':
                stream_mode = True
                print("✅ 已切换到流式模式")
                continue
            
            if user_input.lower() == 'normal':
                stream_mode = False
                print("✅ 已切换到普通模式")
                continue
            
            if user_input.lower() == 'info':
                bot_info = client.get_bot_info()
                if bot_info:
                    print("🤖 智能体信息:")
                    for key, value in bot_info.items():
                        print(f"   {key}: {value}")
                continue
            
            if not user_input:
                print("⚠️ 请输入有效的问题")
                continue
            
            print(f"🤖 用户: {user_input}")
            
            if stream_mode:
                # 流式模式
                print("🤖 智能体: ", end="", flush=True)
                full_response = ""
                for chunk in client.chat_stream(user_input):
                    print(chunk, end="", flush=True)
                    full_response += chunk
                print()  # 换行
            else:
                # 普通模式
                response = client.chat(user_input)
                if response:
                    print(f"🤖 智能体: {response}")
                else:
                    print("❌ 获取回复失败，请重试")
        
        except KeyboardInterrupt:
            print("\n👋 程序被用户中断，再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")


def main():
    """主函数"""
    interactive_chat()


if __name__ == "__main__":
    main() 