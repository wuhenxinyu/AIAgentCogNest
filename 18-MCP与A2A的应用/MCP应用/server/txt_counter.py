"""
@mcp.tool() 注解告诉 MCP 该函数可被 AI 调用。
mcp.run() 启动服务器，可选择stdio、sse或 http 等通信方式。
"""
import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# 创建 MCP Server
mcp = FastMCP("桌面 TXT 文件统计器")

@mcp.tool()
def count_desktop_txt_files() -> int:
    """统计桌面上 .txt 文件的数量"""
    # 获取桌面路径
    desktop_path = Path(os.path.expanduser("/Users/clz/Downloads/xmkf/llm/AIAgentCogNest/18-MCP与A2A的应用/MCP应用/server"))

    # 统计 .txt 文件
    txt_files = list(desktop_path.glob("*.txt"))
    return len(txt_files)

@mcp.tool()
def list_desktop_txt_files() -> str:
    """获取桌面上所有 .txt 文件的列表"""
    # 获取桌面路径
    desktop_path = Path(os.path.expanduser("/Users/clz/Downloads/xmkf/llm/AIAgentCogNest/18-MCP与A2A的应用/MCP应用/server"))

    # 获取所有 .txt 文件
    txt_files = list(desktop_path.glob("*.txt"))

    # 返回文件名列表
    if not txt_files:
        return "桌面上没有找到 .txt 文件。"

    # 格式化文件名列表
    file_list = "\n".join([f"- {file.name}" for file in txt_files])
    return f"在桌面上找到 {len(txt_files)} 个 .txt 文件：\n{file_list}"

@mcp.tool()
def read_txt_file(filename: str) -> str:
    """读取指定txt文件的内容
    
    Args:
        filename: txt文件的名称（例如：test.txt）
        
    Returns:
        文件内容，如果文件不存在则返回错误信息
    """
    # 获取桌面路径
    desktop_path = Path(os.path.expanduser("/Users/clz/Downloads/xmkf/llm/AIAgentCogNest/18-MCP与A2A的应用/MCP应用/server"))
    file_path = desktop_path / filename
    
    # 检查文件是否存在
    if not file_path.exists():
        return f"错误：文件 '{filename}' 不存在于桌面上。"
    
    # 检查文件是否是txt文件
    if file_path.suffix.lower() != '.txt':
        return f"错误：文件 '{filename}' 不是txt文件。"
    
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"文件 '{filename}' 的内容：\n\n{content}"
    except Exception as e:
        return f"读取文件时发生错误：{str(e)}"

if __name__ == "__main__":
    # 初始化并运行服务器，默认stdio方式
    mcp.run(transport="stdio")