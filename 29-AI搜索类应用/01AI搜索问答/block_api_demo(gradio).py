import gradio as gr

with gr.Blocks() as demo:
    with gr.Row():  # 水平行
        with gr.Column():  # 左侧列
            input_text = gr.Textbox(label="输入")
            submit_btn = gr.Button("提交")
        with gr.Column():  # 右侧列
            output_text = gr.Textbox(label="输出")
    submit_btn.click(fn=lambda x: f"你输入了: {x}", inputs=input_text, outputs=output_text)

demo.launch()