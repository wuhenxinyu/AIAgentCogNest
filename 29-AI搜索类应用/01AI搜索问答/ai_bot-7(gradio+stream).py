"""
AI搜索问答机器人
流式传输，支持实时响应，同时集成 Elasticsearch RAG 知识库和 Tavily 网络搜索工具。
组件化gradio的css样式
"""
import os
import asyncio
from typing import Optional, Generator
from qwen_agent.agents import Assistant
import warnings
import gradio as gr
import time
import base64
os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here'
warnings.filterwarnings("ignore")

def init_agent_service():
    """初始化具备 Elasticsearch RAG 和网络搜索能力的助手服务"""
    
    # 步骤 1: LLM 配置
    llm_cfg = {
        'model': 'qwen-max',
        'model_server': 'dashscope',
        'api_key': os.getenv('DASHSCOPE_API_KEY'),
        'generate_cfg': {
            'top_p': 0.8
        }
    }

    # 步骤 2: RAG 配置
    rag_cfg = {
        "rag_backend": "elasticsearch",
        "es": {
            "host": "http://localhost",
            "port": 9200,
            # "user": "elastic",
            # "password": "euqPcOlHrmW18rtaS-3P",
            "index_name": "my_insurance_docs_index"
        },
        "parser_page_size": 500
    }

    # 步骤 3: 系统指令和工具
    system_instruction = '''你是一个AI助手。
请根据用户的问题，优先利用检索工具从本地知识库中查找最相关的信息。
如果本地知识库没有相关信息，再使用 tavily_search 工具从互联网上搜索，并结合这些信息给出专业、准确的回答。'''

    tools_cfg = [{
        "mcpServers": {
            "tavily-mcp": {
                "command": "npx",
                "args": ["-y", "tavily-mcp@0.1.4"],
                "env": {
                    "TAVILY_API_KEY": os.getenv('TAVILY_API_KEY', "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")
                },
                "disabled": False,
                "autoApprove": []
            }
        }
    }]

    # 获取文件夹下所有文件
    file_dir = os.path.join(os.path.dirname(__file__), 'docs')
    files = []
    if os.path.exists(file_dir):
        for file in os.listdir(file_dir):
            file_path = os.path.join(file_dir, file)
            if os.path.isfile(file_path):
                files.append(file_path)
    print('知识库文件列表:', files)

    # 步骤 4: 创建智能体实例
    bot = Assistant(
        llm=llm_cfg,
        system_message=system_instruction,
        function_list=tools_cfg,
        files=files,
        rag_cfg=rag_cfg
    )
    return bot

# 全局变量
bot = init_agent_service()
session_histories = {}

def get_session_id():
    return str(time.time())

def stream_predict(query: str, history: list, session_id: str) -> Generator:
    """Gradio 的核心预测函数 - 支持流式响应"""
    if session_id not in session_histories:
        session_histories[session_id] = []
    
    messages = session_histories[session_id]
    messages.append({'role': 'user', 'content': query})
    
    history[-1][1] = ""
    full_response = ""

    for response in bot.run(messages=messages):
        if response and response[-1]['role'] == 'assistant':
            new_text = response[-1]['content']
            if new_text != full_response:
                delta = new_text[len(full_response):]
                history[-1][1] += delta
                full_response = new_text
                yield history

    messages.append({'role': 'assistant', 'content': full_response})
    session_histories[session_id] = messages


def get_image_base64(image_path):
    """将图片文件转换为 Base64 编码的字符串"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error reading image file: {e}")
        return ""
        
def load_css(css_path):
    """读取 CSS 文件内容"""
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading css file: {e}")
        return ""

def main():
    """启动自定义的 Gradio Web 图形界面"""

    current_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir_path = os.path.join(current_dir, "static")
    
    # --- Base64 嵌入图片 ---
    logo_path = os.path.join(static_dir_path, "logo.png")
    logo_base64 = get_image_base64(logo_path)
    logo_data_uri = f"data:image/png;base64,{logo_base64}"

    # --- 读取并内联 CSS ---
    css_content = load_css(os.path.join(static_dir_path, "styles.css"))
    # 为 Logo 添加大小限制
    css_content += "\n#logo-img { width: 40px !important; height: 40px !important; }"

    app_head = f"""
    <head>
        <link rel="stylesheet" href="file=styles.css">
    </head>
    """
    with gr.Blocks(css=css_content, theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple")) as demo:
        session_id = gr.State(get_session_id)
        
        with gr.Row():
            with gr.Column(scale=2, elem_id="sidebar"):
                with gr.Row(elem_id="logo"):
                    # 使用 Base64 Data URI 直接嵌入 Logo
                    gr.HTML(f'<img id="logo-img" src="{logo_data_uri}" alt="logo">')
                    gr.HTML('<h1 id="logo-text">AI直答</h1>')
                
                gr.Button("🔍  搜索", elem_classes=["sidebar-btn", "active"])
                knowledge_btn = gr.Button("📚  知识库", elem_classes="sidebar-btn")
                favorites_btn = gr.Button("⭐  收藏", elem_classes="sidebar-btn")
                history_btn = gr.Button("🕒  历史", elem_classes="sidebar-btn")
            
            with gr.Column(scale=8, elem_id="main-chat"):
                with gr.Row(elem_id="chat-header"):
                    gr.HTML('<h1 id="chat-header-title">用提问发现世界</h1><p id="chat-header-subtitle">输入你的问题，或使用「@快捷引用」对答主、知识库进行提问</p>')

                chatbot = gr.Chatbot(elem_id="chatbot", bubble_full_width=False, height=450)
                
                with gr.Row(elem_id="suggestion-row") as suggestion_row:
                    suggestions = ['介绍下雇主责任险', '雇主责任险和工伤保险有什么主要区别？', '最近有什么新的保险产品推荐吗？']
                    suggestion_btns = []
                    for s in suggestions:
                        btn = gr.Button(s, elem_classes="suggestion-btn")
                        suggestion_btns.append(btn)
                
                with gr.Row(elem_id="input-container-wrapper"):
                    with gr.Row(elem_id="input-container"):
                        textbox = gr.Textbox(container=False, show_label=False, placeholder="输入你的问题...", scale=10, elem_classes="custom-textbox")
                        submit_btn = gr.Button("↑", scale=1, min_width=0, variant="primary")
        
        
        def on_submit(query, history):
            history.append([query, None])
            return "", history

        def on_suggestion_click(suggestion, history):
            history.append([suggestion, None])
            return "", history, gr.update(visible=False)
        
        def show_not_implemented_toast():
            gr.Info("功能暂未实现，敬请期待！")

        # 为知识库、收藏和历史按钮绑定点击事件，当点击这些按钮时，调用 show_not_implemented_toast 函数
        # 该函数会显示一个提示信息，告知用户此功能暂未实现
        knowledge_btn.click(show_not_implemented_toast, None, None)
        favorites_btn.click(show_not_implemented_toast, None, None)
        history_btn.click(show_not_implemented_toast, None, None)

        # 为文本框绑定提交事件，当用户在文本框中按下回车键时触发
        # 调用 on_submit 函数处理提交内容，输入参数为文本框和聊天机器人组件，输出更新后的文本框和聊天机器人组件
        # queue=False 表示不将此事件加入队列
        submit_event = textbox.submit(on_submit, [textbox, chatbot], [textbox, chatbot], queue=False)
        # 在提交事件完成后，隐藏建议按钮行
        submit_event.then(lambda: gr.update(visible=False), None, suggestion_row)
        # 在提交事件完成后，调用 stream_predict 函数进行流式预测
        # 输入参数为文本框内容、聊天机器人历史记录和会话 ID，输出更新后的聊天机器人组件
        submit_event.then(stream_predict, [textbox, chatbot, session_id], chatbot)

        # 为提交按钮绑定点击事件，当用户点击提交按钮时触发
        # 调用 on_submit 函数处理提交内容，输入参数为文本框和聊天机器人组件，输出更新后的文本框和聊天机器人组件
        # queue=False 表示不将此事件加入队列
        click_event = submit_btn.click(on_submit, [textbox, chatbot], [textbox, chatbot], queue=False)
        # 在点击事件完成后，隐藏建议按钮行
        click_event.then(lambda: gr.update(visible=False), None, suggestion_row)
        # 在点击事件完成后，调用 stream_predict 函数进行流式预测
        # 输入参数为文本框内容、聊天机器人历史记录和会话 ID，输出更新后的聊天机器人组件
        click_event.then(stream_predict, [textbox, chatbot, session_id], chatbot)

        # 遍历所有建议按钮，为每个按钮绑定点击事件
        for btn in suggestion_btns:
            # 当点击建议按钮时，调用 on_suggestion_click 函数
            # 输入参数为按钮文本和聊天机器人历史记录，输出更新后的文本框、聊天机器人组件和隐藏建议按钮行
            # queue=False 表示不将此事件加入队列
            s_click_event = btn.click(on_suggestion_click, [btn, chatbot], [textbox, chatbot, suggestion_row], queue=False)
            # 在建议按钮点击事件完成后，调用 stream_predict 函数进行流式预测
            # 输入参数为按钮文本、聊天机器人历史记录和会话 ID，输出更新后的聊天机器人组件
            s_click_event.then(stream_predict, [btn, chatbot, session_id], chatbot)


    print("正在启动 AI 助手 Web 界面 (v7)...")
    demo.launch()

if __name__ == '__main__':
    main() 