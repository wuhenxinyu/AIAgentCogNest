FROM python:3.12

# 设置工作目录
WORKDIR /app

# 配置阿里云PyPI镜像源以加速依赖安装
RUN mkdir -p /root/.pip && \
    echo "[global]" > /root/.pip/pip.conf && \
    echo "index-url = https://mirrors.aliyun.com/pypi/simple/" >> /root/.pip/pip.conf && \
    echo "trusted-host = mirrors.aliyun.com" >> /root/.pip/pip.conf

# 复制项目文件到容器中
COPY . /app

# 安装项目依赖
RUN pip install --no-cache-dir dashscope qwen_agent markdown beautifulsoup4 python-docx python-dotenv qwen_agent[gui,mcp,rag]

# 设置环境变量
# ENV DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY}
ENV QWEN_AGENT_DEFAULT_MAX_INPUT_TOKENS=158000
ENV GRADIO_TEMP_DIR=/app/upload

# 暴露7860端口
EXPOSE 7860

# 创建upload目录
RUN mkdir -p /app/upload && \
    chmod -R 777 /app/upload

# 设置启动命令
CMD ["python", "assistant_api_doc_bot.py"]