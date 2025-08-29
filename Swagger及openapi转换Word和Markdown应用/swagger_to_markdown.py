import json
import argparse
import os
from datetime import datetime


def swagger_to_markdown(input_file_path, output_file_path=None):
    """将Swagger/OpenAPI JSON转换为Markdown文档"""
    # 读取Swagger JSON文件
    with open(input_file_path, 'r', encoding='utf-8') as f:
        swagger_data = json.load(f)
    
    # 生成输出文件路径
    if output_file_path is None:
        base_name = os.path.splitext(os.path.basename(input_file_path))[0]
        output_file_path = f"{base_name}_api文档.md"
    
    # 存储Markdown内容
    md_content = []
    
    # 4. 数据模型（提前获取并存储所有数据模型定义）
    # 获取数据模型定义（支持OpenAPI 3.0和Swagger 2.0）
    definitions = swagger_data.get('definitions', {})  # Swagger 2.0
    if not definitions:
        definitions = swagger_data.get('components', {}).get('schemas', {})  # OpenAPI 3.0
    
    # 5. API路径
    paths = swagger_data.get('paths', {})
    if paths:
        # 按路径分组
        for path, path_item in paths.items():
            for method, operation in path_item.items():
                md_content.append(f" ")
                md_content.append(f" ")
                # 方法名称转换为大写
                method = method.upper()
                
                # 获取接口信息
                summary = operation.get('summary', '')
                desc = operation.get('description', '')
                tags = operation.get('tags', [])
                consumes = operation.get('consumes', [])
                produces = operation.get('produces', [])
                
                # 添加接口标题和锚点
                interface_title = summary if summary else f"{method} {path}"
                md_content.append(f"### 名称：{interface_title}")

                # 1. 在读取Swagger数据后获取basePath
                # 读取Swagger JSON文件
                with open(input_file_path, 'r', encoding='utf-8') as f:
                    swagger_data = json.load(f)
                
                # 获取basePath（如果存在）
                base_path = swagger_data.get('basePath', '')
                
                # 2. 在生成请求路径时拼接basePath
                md_content.append(f"- **请求路径**: https://xxx.xxx.xxx.xxxx{base_path}{path}")
                md_content.append(f"- **请求方法**: {method}")
                md_content.append(f"- **接口描述**: {desc if desc else '暂无描述'}")
                
                # 添加标签
                if tags:
                    md_content.append(f"- **接口标签**: {', '.join(tags)}")
                    
                md_content.append("- **请求参数**:")
                # 请求参数
                parameters = operation.get('parameters', [])
                if parameters:
                    # 按参数位置分类
                    query_params = []
                    header_params = []
                    path_params = []
                    body_params = []
                    
                    for param in parameters:
                        param_location = param.get('in', 'query')
                        if param_location == 'query':
                            query_params.append(param)
                        elif param_location == 'header':
                            header_params.append(param)
                        elif param_location == 'path':
                            path_params.append(param)
                        elif param_location == 'body':
                            body_params.append(param)
                    
                    # 显示Query参数
                    if query_params:
                        md_content.append("<table>")
                        md_content.append("  <thead class='ant-table-thead'>")
                        md_content.append("    <tr>")
                        md_content.append("      <th key=name>参数名称</th><th key=type>类型</th><th key=required>是否必须</th><th key=default>默认值</th><th key=example>示例</th><th key=desc>备注</th>")
                        md_content.append("    </tr>")
                        # 在第87行附近修改Query参数表格的tbody属性
                        md_content.append("  </thead><tbody class='ant-table-tbody'>")
                        
                        for idx, param in enumerate(query_params):
                            name = param.get('name', '')
                            required = '是' if param.get('required', False) else '否'
                            example = param.get('example', '') if 'example' in param else ''
                            desc = param.get('description', '')
                            default_value = param.get('default', '')
                            
                            # 获取参数类型和长度信息
                            param_type = param.get('type', 'string')
                            if 'format' in param:
                                param_type = f"{param_type}({param['format']})"
                            
                            # 添加长度信息
                            length_info = []
                            if 'maxLength' in param:
                                length_info.append(f"最大长度: {param['maxLength']}")
                            if 'minLength' in param:
                                length_info.append(f"最小长度: {param['minLength']}")
                            if 'maximum' in param:
                                length_info.append(f"最大值: {param['maximum']}")
                            if 'minimum' in param:
                                length_info.append(f"最小值: {param['minimum']}")
                            
                            # 合并类型和长度信息
                            full_type_info = param_type
                            if length_info:
                                full_type_info += " (" + ", ".join(length_info) + ")"
                            
                            md_content.append(f"    <tr key={idx}>")
                            md_content.append(f"      <td key=0>{name}</td>")
                            md_content.append(f"      <td key=1>{full_type_info}</td>")
                            md_content.append(f"      <td key=2>{required}</td>")
                            md_content.append(f"      <td key=3>{default_value}</td>")
                            md_content.append(f"      <td key=4>{example}</td>")
                            md_content.append(f"      <td key=5>{desc}</td>")
                            md_content.append(f"    </tr>")
                        
                        md_content.append("  </tbody>")
                        md_content.append("</table>")
                    
                    # 显示Header参数
                    if header_params:
                        md_content.append("<table>")
                        md_content.append("  <thead class='ant-table-thead'>")
                        md_content.append("    <tr>")
                        md_content.append("      <th key=name>参数名称</th><th key=type>类型</th><th key=required>是否必须</th><th key=default>默认值</th><th key=example>示例</th><th key=desc>备注</th>")
                        md_content.append("    </tr>")
                        md_content.append("  </thead><tbody className='ant-table-tbody'>")
                        
                        for idx, param in enumerate(header_params):
                            name = param.get('name', '')
                            required = '是' if param.get('required', False) else '否'
                            desc = param.get('description', '')
                            default_value = param.get('default', '')
                            
                            # 获取参数类型和长度信息
                            param_type = param.get('type', 'string')
                            if 'format' in param:
                                param_type = f"{param_type}({param['format']})"
                            
                            # 添加长度信息
                            length_info = []
                            if 'maxLength' in param:
                                length_info.append(f"最大长度: {param['maxLength']}")
                            if 'minLength' in param:
                                length_info.append(f"最小长度: {param['minLength']}")
                            
                            # 合并类型和长度信息
                            full_type_info = param_type
                            if length_info:
                                full_type_info += " (" + ", ".join(length_info) + ")"
                            
                            md_content.append(f"    <tr key={idx}>")
                            md_content.append(f"      <td key=0>{name}</td>")
                            md_content.append(f"      <td key=1>{full_type_info}</td>")
                            md_content.append(f"      <td key=2>{required}</td>")
                            md_content.append(f"      <td key=3>{default_value}</td>")
                            md_content.append(f"      <td key=4>{example}</td>")
                            md_content.append(f"      <td key=5>{desc}</td>")
                            md_content.append(f"    </tr>")
                        
                        md_content.append("  </tbody>")
                        md_content.append("</table>")
                    
                    # 显示Path参数
                    if path_params:
                        md_content.append("<table>")
                        md_content.append("  <thead class='ant-table-thead'>")
                        md_content.append("    <tr>")
                        md_content.append("      <th key=name>参数名称</th><th key=type>类型</th><th key=required>是否必须</th><th key=default>默认值</th><th key=example>示例</th><th key=desc>备注</th>")
                        md_content.append("    </tr>")
                        md_content.append("  </thead><tbody className='ant-table-tbody'>")
                        
                        for idx, param in enumerate(path_params):
                            name = param.get('name', '')
                            required = '是' if param.get('required', False) else '否'
                            example = param.get('example', '') if 'example' in param else ''
                            desc = param.get('description', '')
                            default_value = param.get('default', '')
                            
                            # 获取参数类型和长度信息
                            param_type = param.get('type', 'string')
                            if 'format' in param:
                                param_type = f"{param_type}({param['format']})"
                            
                            # 添加长度信息
                            length_info = []
                            if 'maxLength' in param:
                                length_info.append(f"最大长度: {param['maxLength']}")
                            if 'minLength' in param:
                                length_info.append(f"最小长度: {param['minLength']}")
                            if 'maximum' in param:
                                length_info.append(f"最大值: {param['maximum']}")
                            if 'minimum' in param:
                                length_info.append(f"最小值: {param['minimum']}")
                            
                            # 合并类型和长度信息
                            full_type_info = param_type
                            if length_info:
                                full_type_info += " (" + ", ".join(length_info) + ")"
                            
                            md_content.append(f"    <tr key={idx}>")
                            md_content.append(f"      <td key=0>{name}</td>")
                            md_content.append(f"      <td key=1>{full_type_info}</td>")
                            md_content.append(f"      <td key=2>{required}</td>")
                            md_content.append(f"      <td key=3>{default_value}</td>")
                            md_content.append(f"      <td key=4>{example}</td>")
                            md_content.append(f"      <td key=5>{desc}</td>")
                            md_content.append(f"    </tr>")
                        
                        md_content.append("  </tbody>")
                        md_content.append("</table>")
                    
                    # 显示Body参数
                    if body_params:
                        md_content.append("<table>")
                        md_content.append("  <thead class='ant-table-thead'>")
                        md_content.append("    <tr>")
                        md_content.append("      <th key=name>参数名称</th><th key=type>类型</th><th key=required>是否必须</th><th key=default>默认值</th><th key=desc>备注</th>")
                        md_content.append("    </tr>")
                        # 在第243行附近修改Body参数表格的tbody属性
                        md_content.append("  </thead><tbody class='ant-table-tbody'>")
                        
                        for idx, param in enumerate(body_params):
                            name = param.get('name', '')
                            required = '是' if param.get('required', False) else '否'
                            desc = param.get('description', '')
                            default_value = param.get('default', '')
                            
                            # 获取参数类型
                            param_type = 'object'
                            if 'schema' in param:
                                schema = param['schema']
                                # 1. 在处理Body参数的schema引用时添加替换
                                if '$ref' in schema:
                                    ref_name = schema['$ref'].split('/')[-1]
                                    # 添加替换逻辑
                                    ref_name = ref_name.replace('devops_aishu_cn_AISHUDevOps_AnyFabric__git_', '')
                                    # 检查引用的实际类型
                                    if ref_name in definitions:
                                        ref_schema = definitions[ref_name]
                                        actual_type = ref_schema.get('type', 'object')
                                        param_type = f"{ref_name} ({actual_type})"
                                    else:
                                        param_type = ref_name

                                    # 2. 在处理请求参数结构体详情时添加替换 - 修复：缩进到if '$ref' in schema条件内
                                    ref_name = param['schema']['$ref'].split('/')[-1]
                                    # 添加替换逻辑
                                    ref_name = ref_name.replace('devops_aishu_cn_AISHUDevOps_AnyFabric__git_', '')
                                    if ref_name in definitions:
                                        schema = definitions[ref_name]
                                        md_content.append(f"\n- **请求参数结构体详情**:")
                                        md_content.append("<table>")
                                        md_content.append("  <thead class='ant-table-thead'>")
                                        md_content.append("    <tr>")
                                        md_content.append("      <th key=name>字段名称</th><th key=type>类型</th><th key=required>是否必须</th><th key=default>默认值</th><th key=desc>备注</th>")
                                        md_content.append("    </tr>")
                                        # 在第274行附近修改请求参数结构体详情表格的tbody属性
                                        md_content.append("  </thead><tbody class='ant-table-tbody'>")
                                        
                                        # 使用expand_schema_as_tree函数展开结构体字段
                                        expand_schema_as_tree(md_content, schema, definitions, ref_name)
                                    
                                    md_content.append("  </tbody>")
                                    md_content.append("</table>")
                                    md_content.append("")
                
                # 请求体
                request_body = operation.get('requestBody', {})
                if request_body:
                    if 'content' in request_body:
                        for content_type, content in request_body['content'].items():
                            md_content.append(f"**Content-Type**: {content_type}")
                            md_content.append("<table>")
                            md_content.append("  <thead class='ant-table-thead'>")
                            md_content.append("    <tr>")
                            md_content.append("      <th key=name>名称</th><th key=type>类型</th><th key=required>是否必须</th><th key=default>默认值</th><th key=desc>备注</th>")
                            md_content.append("    </tr>")
                            md_content.append("  </thead><tbody className='ant-table-tbody'>")
                            
                            if 'schema' in content:
                                schema = content['schema']
                                if '$ref' in schema:
                                    ref_name = schema['$ref'].split('/')[-1]
                                    if ref_name in definitions:
                                        schema = definitions[ref_name]
                                        # 展开schema结构
                                        expand_schema_as_tree(md_content, schema, definitions, ref_name)
                                else:
                                    # 展开简单schema结构
                                    expand_schema_as_tree(md_content, schema, definitions, "body")
                            
                            md_content.append("  </tbody>")
                            md_content.append("</table>")
                
                # 响应参数
                responses = operation.get('responses', {})
                if responses:
                    md_content.append("- **响应参数**:")
                    
                    # 只选择主要的成功响应来展开（通常是200）
                    primary_response = None
                    for status_code, response in responses.items():
                        if status_code.startswith('2'):  # 成功状态码
                            primary_response = response
                            break
                        
                    # 修复缩进：将if条件移到for循环外部
                    if primary_response:
                        md_content.append("<table>")
                        md_content.append("  <thead class='ant-table-thead'>")
                        md_content.append("    <tr>")
                        md_content.append("      <th key=name>名称</th><th key=type>类型</th><th key=required>是否必须</th><th key=default>默认值</th><th key=desc>备注</th>")
                        md_content.append("    </tr>")
                        md_content.append("  </thead><tbody className='ant-table-tbody'>")
                        
                        if 'content' in primary_response:
                            for content_type, content in primary_response['content'].items():
                                if 'schema' in content:
                                    schema = content['schema']
                                    if '$ref' in schema:
                                        ref_name = schema['$ref'].split('/')[-1]
                                        if ref_name in definitions:
                                            schema = definitions[ref_name]
                                            # 修复：移除is_response_param参数
                                            expand_schema_as_tree(md_content, schema, definitions, ref_name, "", 0, 0)
                                    else:
                                        # 修复：移除is_response_param参数
                                        expand_schema_as_tree(md_content, schema, definitions, "response", "", 0, 0)
                        elif 'schema' in primary_response and primary_response['schema']:
                                # Swagger 2.0格式
                                # 修复：首先从primary_response中获取schema
                                schema = primary_response['schema']
                                
                                # 检查schema是否包含引用
                                if '$ref' in schema:
                                    ref_name = schema['$ref'].split('/')[-1]
                                    if ref_name in definitions:
                                        schema = definitions[ref_name]
                                        # 展开schema结构
                                        expand_schema_as_tree(md_content, schema, definitions, ref_name, "", 0, 0)
                                else:
                                    # 展开简单schema结构
                                    expand_schema_as_tree(md_content, schema, definitions, "response", "", 0, 0)
                        
                        md_content.append("  </tbody>")
                        md_content.append("</table>")
                    
                    # 显示所有状态码的简要说明
                    status_codes_table = False
                    for status_code, response in responses.items():
                        if not status_codes_table:
                            md_content.append("- **状态码说明**")
                            md_content.append("<table>")
                            md_content.append("  <thead class='ant-table-thead'>")
                            md_content.append("    <tr>")
                            md_content.append("      <th key=code>状态码</th><th key=desc>描述</th>")
                            md_content.append("    </tr>")
                            md_content.append("  </thead><tbody className='ant-table-tbody'>")
                            status_codes_table = True
                        
                        description = response.get('description', '')
                        md_content.append(f"    <tr key={status_code}>")
                        md_content.append(f"      <td key=code>{status_code}</td>")
                        md_content.append(f"      <td key=desc>{description}</td>")
                        md_content.append(f"    </tr>")
                    
                    if status_codes_table:
                        md_content.append("  </tbody>")
                        md_content.append("</table>")
                
                md_content.append("")
    
    # 将Markdown内容写入文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_content))
    
    print(f"Markdown文档已生成: {output_file_path}")


def expand_schema_as_tree(md_content, schema, definitions, schema_name=None, parent_path="", indent_level=0, row_index=0):
    """将schema展开为树形结构的HTML表格行"""
    # 处理数组类型
    if schema.get('type') == 'array' and 'items' in schema:
        items_schema = schema['items']
        item_type = 'object'
        
        if '$ref' in items_schema:
            ref_name = items_schema['$ref'].split('/')[-1]
            item_type = ref_name
            if ref_name in definitions:
                items_schema = definitions[ref_name]
        elif 'type' in items_schema:
            item_type = items_schema['type']
            if 'format' in items_schema:
                item_type = f"{item_type}({items_schema['format']})"
        
        # 显示数组行
        padding = indent_level * 20
        md_content.append(f"    <tr key={row_index}-{indent_level}>")
        md_content.append(f'      <td key=0><span style="padding-left: {padding}px"><span style="color: #8c8a8a"></span> {schema_name}</span></td>')
        md_content.append(f"      <td key=1><span>{schema.get('type')} [{item_type}]</span></td>")
        md_content.append(f"      <td key=2>是</td>")
        md_content.append(f"      <td key=3></td>")
        md_content.append(f'      <td key=4><span style="white-space: pre-wrap">{schema.get("description", "")}</span></td>')
        md_content.append(f"    </tr>")
        
        # 如果items是对象或有引用，递归展开（添加深度限制）
        if ((items_schema.get('type') == 'object' and 'properties' in items_schema) or 
            ('$ref' in items_schema and items_schema['$ref'] in definitions)) and indent_level < 3:  # 添加深度限制
            expand_schema_as_tree(md_content, items_schema, definitions, "", f"{parent_path}.{schema_name}", indent_level + 1, row_index)
        
        return
    
    # 处理对象类型
    if schema.get('type') == 'object' and 'properties' in schema:
        required_fields = schema.get('required', [])
        
        # 如果是顶层对象，不显示行，直接展开属性
        if indent_level == 0:
            for prop_name, prop_schema in schema['properties'].items():
                expand_schema_as_tree(md_content, prop_schema, definitions, prop_name, parent_path, indent_level, row_index)
        else:
            # 如果是嵌套对象，显示对象行并递归展开属性
            padding = indent_level * 20
            md_content.append(f"    <tr key={row_index}-{indent_level}>")
            md_content.append(f'      <td key=0><span style="padding-left: {padding}px"><span style="color: #8c8a8a">├─</span> {schema_name}</span></td>')
            md_content.append(f"      <td key=1><span>object</span></td>")
            md_content.append(f"      <td key=2>{'是' if schema_name in required_fields else '否'}</td>")
            md_content.append(f"      <td key=3></td>")
            md_content.append(f'      <td key=4><span style="white-space: pre-wrap">{schema.get("description", "")}</span></td>')
            md_content.append(f"    </tr>")
            
            # 递归展开属性（添加深度限制）
            if indent_level < 3:  # 添加深度限制
                for prop_name, prop_schema in schema['properties'].items():
                    expand_schema_as_tree(md_content, prop_schema, definitions, prop_name, f"{parent_path}.{schema_name}", indent_level + 1, row_index)
        
        return
    
    # 处理引用类型
    if '$ref' in schema:
        ref_name = schema['$ref'].split('/')[-1]
        padding = indent_level * 20
        
        # 显示引用行
        md_content.append(f"    <tr key={row_index}-{indent_level}>")
        md_content.append(f'      <td key=0><span style="padding-left: {padding}px"><span style="color: #8c8a8a">{'' if indent_level == 0 else '├─'}</span> {schema_name}</span></td>')
        md_content.append(f'      <td key=1><span>{ref_name}</span></td>')
        md_content.append(f"      <td key=2>{'是' if schema_name in (schema.get('required', [])) else '否'}</td>")
        md_content.append(f"      <td key=3>{schema.get('default', '')}</td>")
        md_content.append(f'      <td key=4><span style="white-space: pre-wrap">{schema.get("description", "")}</span></td>')
        md_content.append(f"    </tr>")
        
        # 如果引用存在于definitions中，递归展开
        if ref_name in definitions and indent_level < 3:  # 限制递归深度
            ref_schema = definitions[ref_name]
            expand_schema_as_tree(md_content, ref_schema, definitions, "", f"{parent_path}.{schema_name}", indent_level + 1, row_index)
        
        return
    
    # 处理基本类型
    prop_type = schema.get('type', '未知')
    if 'format' in schema:
        prop_type = f"{prop_type}({schema['format']})"
    
    padding = indent_level * 20
    
    # 显示基本类型行
    md_content.append(f"    <tr key={row_index}-{indent_level}>")
    md_content.append(f'      <td key=0><span style="padding-left: {padding}px"><span style="color: #8c8a8a">{'' if indent_level == 0 else '├─'}</span> {schema_name}</span></td>')
    md_content.append(f'      <td key=1><span>{prop_type}</span></td>')
    md_content.append(f"      <td key=2>{'是' if schema_name in (schema.get('required', [])) else '否'}</td>")
    md_content.append(f"      <td key=3>{schema.get('default', '')}</td>")
    md_content.append(f'      <td key=4><span style="white-space: pre-wrap">{schema.get("description", "")}</span></td>')
    md_content.append(f"    </tr>")


def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description='将Swagger/OpenAPI JSON转换为Markdown文档')
    parser.add_argument('input_file', help='输入的Swagger/OpenAPI JSON文件路径')
    parser.add_argument('-o', '--output', help='输出的Markdown文件路径', default=None)
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件 '{args.input_file}' 不存在")
        return
    
    # 执行转换
    swagger_to_markdown(args.input_file, args.output)


if __name__ == '__main__':
    main()