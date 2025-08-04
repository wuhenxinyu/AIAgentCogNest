"""
1、告警内容理解。根据输入的告警信息，结合第三方接口数据，判断当前的异常情况（告警对象、异常模式）；
2、分析方法建议。根据当前告警内容，结合应急预案、运维文档和大语言模型自有知识，形成分析方法的建议；
3、分析内容自动提取。根据用户输入的分析内容需求，调用多种第三方接口获取分析数据，并进行总结；
4、处置方法推荐和执行。根据当前上下文的故障场景理解，结合应急预案和第三方接口，形成推荐处置方案，待用户确认后调用第三方接口进行执行。
"""
import json
import os
import random
import time
import dashscope
from dashscope.api_entities.dashscope_response import Role
from dashscope.common.constants import DASHSCOPE_API_KEY_ENV
# 阿里云百炼智能开放平台申请API Key
os.environ[DASHSCOPE_API_KEY_ENV] = "your_api_key_here"
# 从环境变量中，获取 DASHSCOPE_API_KEY
api_key = os.environ.get(DASHSCOPE_API_KEY_ENV)
dashscope.api_key = api_key

# 通过第三方接口获取数据库服务器状态
def get_current_status():
    # 生成连接数数据
    connections = random.randint(10, 100)
    # 生成CPU使用率数据
    cpu_usage = round(random.uniform(1, 100), 1)
    # 生成内存使用率数据
    memory_usage = round(random.uniform(10, 100), 1)
    status_info = {
        "连接数": connections,
        "CPU使用率": f"{cpu_usage}%",
        "内存使用率": f"{memory_usage}%"
    }
    return json.dumps(status_info, ensure_ascii=False)

# 推荐处置方案
def recommend_solutions(alert_type, current_status=None):
    # 根据告警类型和当前状态推荐处置方案
    solutions = []
    
    if "数据库连接数" in alert_type:
        solutions = [
            {
                "id": "1",
                "name": "优化数据库连接池",
                "description": "调整数据库连接池参数，增加最大连接数并优化连接回收机制",
                "steps": [
                    "检查当前连接池配置",
                    "调整最大连接数参数",
                    "优化连接超时设置",
                    "重启应用服务"
                ],
                "risk_level": "低"
            },
            {
                "id": "2",
                "name": "识别并关闭空闲连接",
                "description": "查找并关闭长时间空闲的数据库连接，释放资源",
                "steps": [
                    "执行数据库查询识别空闲连接",
                    "关闭超过30分钟的空闲连接",
                    "记录关闭的连接数量"
                ],
                "risk_level": "低"
            },
            {
                "id": "3",
                "name": "扩展数据库资源",
                "description": "增加数据库服务器资源或启动读写分离，分散连接压力",
                "steps": [
                    "评估当前数据库负载",
                    "准备额外的数据库实例",
                    "配置读写分离",
                    "迁移部分连接到新实例"
                ],
                "risk_level": "中"
            }
        ]
    elif "CPU使用率" in alert_type:
        solutions = [
            {
                "id": "1",
                "name": "识别高CPU进程",
                "description": "查找并分析导致CPU高使用率的进程",
                "steps": [
                    "执行top命令查看进程CPU使用情况",
                    "分析高CPU使用率进程的功能和重要性",
                    "检查是否存在异常或恶意进程"
                ],
                "risk_level": "低"
            },
            {
                "id": "2",
                "name": "优化或重启高负载应用",
                "description": "对高CPU使用的应用进行优化或重启",
                "steps": [
                    "备份应用当前状态和数据",
                    "优化应用配置或代码",
                    "必要时重启应用",
                    "监控重启后的CPU使用情况"
                ],
                "risk_level": "中"
            }
        ]
    elif "内存使用率" in alert_type:
        solutions = [
            {
                "id": "1",
                "name": "识别内存泄漏",
                "description": "检查是否存在内存泄漏问题并定位",
                "steps": [
                    "使用内存分析工具检查进程内存使用",
                    "识别可能存在内存泄漏的应用",
                    "分析内存使用增长趋势"
                ],
                "risk_level": "低"
            },
            {
                "id": "2",
                "name": "清理系统缓存",
                "description": "清理系统缓存释放内存空间",
                "steps": [
                    "执行缓存清理命令",
                    "监控内存释放情况",
                    "评估清理效果"
                ],
                "risk_level": "低"
            }
        ]
    else:
        solutions = [
            {
                "id": "1",
                "name": "通用系统检查",
                "description": "执行通用系统检查流程，排查常见问题",
                "steps": [
                    "检查系统日志文件",
                    "检查关键服务状态",
                    "检查网络连接情况",
                    "检查存储空间使用情况"
                ],
                "risk_level": "低"
            }
        ]
    
    return json.dumps({"solutions": solutions}, ensure_ascii=False)

# 执行处置方案
def execute_solution(solution_id, alert_type):
    # 模拟执行处置方案
    execution_result = {
        "status": "success",
        "execution_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "details": []
    }
    
    if "数据库连接数" in alert_type:
        if solution_id == "1":
            execution_result["details"] = [
                "已检查连接池配置，当前最大连接数为50",
                "已将最大连接数调整为100",
                "已将连接超时时间从30分钟调整为15分钟",
                "已重启应用服务，新配置生效"
            ]
        elif solution_id == "2":
            idle_connections = random.randint(5, 20)
            execution_result["details"] = [
                f"已执行数据库查询，发现{idle_connections}个空闲连接",
                f"已成功关闭{idle_connections}个空闲连接",
                "系统连接数已降低，继续监控中"
            ]
        elif solution_id == "3":
            execution_result["details"] = [
                "已评估数据库负载，当前单实例压力较大",
                "已准备额外的读库实例",
                "已配置读写分离，读操作将转发到新实例",
                "已成功迁移30%的连接到新实例"
            ]
    elif "CPU使用率" in alert_type:
        if solution_id == "1":
            execution_result["details"] = [
                "已执行top命令，发现Java进程占用CPU 85%",
                "分析显示该进程为核心业务应用，非异常进程",
                "建议优化应用代码或增加服务器资源"
            ]
        elif solution_id == "2":
            execution_result["details"] = [
                "已备份应用当前状态和关键数据",
                "已优化JVM参数配置",
                "已重启应用服务",
                "重启后CPU使用率降至45%，继续监控中"
            ]
    elif "内存使用率" in alert_type:
        if solution_id == "1":
            execution_result["details"] = [
                "已使用内存分析工具检查进程",
                "发现NodeJS应用存在内存持续增长现象",
                "分析显示可能存在内存泄漏，建议联系开发团队修复"
            ]
        elif solution_id == "2":
            freed_memory = random.randint(500, 2000)
            execution_result["details"] = [
                "已执行系统缓存清理命令",
                f"已释放{freed_memory}MB内存空间",
                "内存使用率降至65%，继续监控中"
            ]
    else:
        execution_result["details"] = [
            "已检查系统日志，未发现异常记录",
            "已确认所有关键服务运行正常",
            "已检查网络连接，网络状态良好",
            "已检查存储空间，使用率正常"
        ]
    
    return json.dumps(execution_result, ensure_ascii=False)

# 封装模型响应函数
def get_response(messages):
    # 调用通义千问大模型API
    response = dashscope.Generation.call(
        model='qwen-turbo',        # 使用通义千问大模型,用于理解告警内容并提供处置建议
        messages=messages,         # 传入对话历史消息
        tools=tools,              # 传入可调用的工具函数列表
        result_format='message'    # 将输出设置为message形式
    )
    return response
    
# 获取当前局部命名空间中的所有变量和函数，用于后续动态调用函数
current_locals = locals()
print('current_locals=', current_locals)

tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_status",
                "description": "调用监控系统接口，获取当前数据库服务器性能指标，包括：连接数、CPU使用率、内存使用率",
                "parameters": {
                },
                "required": []
            }                
        },
        {
            "type": "function",
            "function": {
                "name": "recommend_solutions",
                "description": "根据告警类型和当前系统状态，推荐可行的处置方案列表",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "alert_type": {
                            "type": "string",
                            "description": "告警类型，例如：数据库连接数超过阈值、CPU使用率过高、内存使用率过高等"
                        },
                        "current_status": {
                            "type": "string",
                            "description": "当前系统状态的JSON字符串，包含各项指标数据"
                        }
                    },
                    "required": ["alert_type"]
                }
            }                
        },
        {
            "type": "function",
            "function": {
                "name": "execute_solution",
                "description": "执行用户确认的处置方案",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "solution_id": {
                            "type": "string",
                            "description": "处置方案ID"
                        },
                        "alert_type": {
                            "type": "string",
                            "description": "告警类型，用于确定执行上下文"
                        }
                    },
                    "required": ["solution_id", "alert_type"]
                }
            }                
        }
    ]

query = """告警：数据库连接数超过设定阈值
时间：2024-08-03 15:30:00
"""
messages=[
    {"role": "system", "content": "我是运维分析师，用户会告诉我们告警内容。我会基于告警内容，判断当前的异常情况（告警对象、异常模式），获取系统状态，推荐处置方案，并在用户确认后执行处置方案。"},
    {"role": "user", "content": query}]


while True:
    response = get_response(messages)
    message = response.output.choices[0].message
    messages.append(message)
    print('response=', response)

    if response.output.choices[0].finish_reason == 'stop':
        break
    
    # 判断用户是否要call function
    if message.tool_calls:
        # 获取fn_name, fn_arguments
        fn_name = message.tool_calls[0]['function']['name']
        fn_arguments = message.tool_calls[0]['function']['arguments']
        arguments_json = json.loads(fn_arguments)
        print(f'fn_name={fn_name} fn_arguments={fn_arguments}')
        function = current_locals[fn_name]
        tool_response = function(**arguments_json)
        tool_info = {"name": fn_name, "role":"tool", "content": tool_response}
        messages.append(tool_info)

print('messages=', messages)

# 示例用户确认执行处置方案的交互
def simulate_user_confirmation():
    print("\n=== 模拟用户确认处置方案 ===")
    print("请输入要执行的处置方案ID（或输入'q'退出）:")
    solution_id = input()
    if solution_id.lower() == 'q':
        return False
    
    # 添加用户确认消息
    user_confirmation = f"我确认执行ID为{solution_id}的处置方案"
    messages.append({"role": "user", "content": user_confirmation})
    
    # 继续对话流程
    while True:
        response = get_response(messages)
        message = response.output.choices[0].message
        messages.append(message)
        print('response=', response)

        if response.output.choices[0].finish_reason == 'stop':
            break
        
        # 判断用户是否要call function
        if message.tool_calls:
            # 获取fn_name, fn_arguments
            fn_name = message.tool_calls[0]['function']['name']
            fn_arguments = message.tool_calls[0]['function']['arguments']
            arguments_json = json.loads(fn_arguments)
            print(f'fn_name={fn_name} fn_arguments={fn_arguments}')
            function = current_locals[fn_name]
            tool_response = function(**arguments_json)
            tool_info = {"name": fn_name, "role":"tool", "content": tool_response}
            messages.append(tool_info)
    
    return True

# 在主流程结束后，可以调用此函数模拟用户确认
if __name__ == "__main__":
    # 先执行主流程
    print("\n=== 主流程执行完毕，是否模拟用户确认处置方案？(y/n) ===")
    choice = input()
    if choice.lower() == 'y':
        while simulate_user_confirmation():
            print("\n=== 是否继续执行其他处置方案？(y/n) ===")
            choice = input()
            if choice.lower() != 'y':
                break

