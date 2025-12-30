import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import json
import io
import base64
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DataVisualizer:
    """数据可视化工具类"""

    @staticmethod
    def create_chart(data: List[Dict[str, Any]], chart_type: str,
                    x_column: Optional[str] = None,
                    y_column: Optional[str] = None,
                    group_by: Optional[str] = None,
                    title: Optional[str] = None) -> Dict[str, Any]:
        """
        创建图表

        Args:
            data: 数据列表
            chart_type: 图表类型
            x_column: X轴列名
            y_column: Y轴列名
            group_by: 分组列名
            title: 图表标题

        Returns:
            包含图表HTML或JSON的字典
        """
        try:
            # 转换为DataFrame
            df = pd.DataFrame(data)

            if df.empty:
                return {"success": False, "error": "No data to visualize"}

            # 根据图表类型创建图表
            if chart_type == "bar":
                result = DataVisualizer._create_bar_chart(df, x_column, y_column, group_by, title)
            elif chart_type == "line":
                result = DataVisualizer._create_line_chart(df, x_column, y_column, group_by, title)
            elif chart_type == "pie":
                result = DataVisualizer._create_pie_chart(df, x_column, y_column, title)
            elif chart_type == "scatter":
                result = DataVisualizer._create_scatter_chart(df, x_column, y_column, group_by, title)
            elif chart_type == "histogram":
                result = DataVisualizer._create_histogram(df, x_column, title)
            elif chart_type == "box":
                result = DataVisualizer._create_box_chart(df, x_column, y_column, group_by, title)
            elif chart_type == "heatmap":
                result = DataVisualizer._create_heatmap(df, title)
            else:
                result = {"success": False, "error": f"Unsupported chart type: {chart_type}"}

            return result

        except Exception as e:
            logger.error(f"Error creating chart: {str(e)}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def _create_bar_chart(df: pd.DataFrame, x_column: Optional[str],
                         y_column: Optional[str], group_by: Optional[str],
                         title: Optional[str]) -> Dict[str, Any]:
        """创建柱状图"""
        try:
            if not x_column:
                x_column = df.columns[0]
            if not y_column:
                # 选择第一个数值列
                numeric_cols = df.select_dtypes(include=['number']).columns
                y_column = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[1]

            fig = px.bar(df, x=x_column, y=y_column, color=group_by,
                        title=title or f"{y_column} by {x_column}")

            fig.update_layout(
                xaxis_title=x_column,
                yaxis_title=y_column,
                hovermode='x unified'
            )

            return {
                "success": True,
                "chart_html": fig.to_html(include_plotlyjs='cdn'),
                "chart_json": json.dumps(fig, cls=PlotlyJSONEncoder)
            }

        except Exception as e:
            logger.error(f"Error creating bar chart: {str(e)}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def _create_line_chart(df: pd.DataFrame, x_column: Optional[str],
                          y_column: Optional[str], group_by: Optional[str],
                          title: Optional[str]) -> Dict[str, Any]:
        """创建折线图"""
        try:
            if not x_column:
                x_column = df.columns[0]
            if not y_column:
                numeric_cols = df.select_dtypes(include=['number']).columns
                y_column = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[1]

            fig = px.line(df, x=x_column, y=y_column, color=group_by,
                         title=title or f"{y_column} over {x_column}")

            fig.update_layout(
                xaxis_title=x_column,
                yaxis_title=y_column,
                hovermode='x unified'
            )

            return {
                "success": True,
                "chart_html": fig.to_html(include_plotlyjs='cdn'),
                "chart_json": json.dumps(fig, cls=PlotlyJSONEncoder)
            }

        except Exception as e:
            logger.error(f"Error creating line chart: {str(e)}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def _create_pie_chart(df: pd.DataFrame, x_column: Optional[str],
                         y_column: Optional[str], title: Optional[str]) -> Dict[str, Any]:
        """创建饼图"""
        try:
            if not x_column:
                x_column = df.columns[0]

            # 如果指定了y_column，先聚合
            if y_column and y_column != x_column:
                df_agg = df.groupby(x_column)[y_column].sum().reset_index()
            else:
                df_agg = df[x_column].value_counts().reset_index()
                df_agg.columns = [x_column, 'value']

            fig = px.pie(df_agg, values='value', names=x_column,
                        title=title or f"Distribution of {x_column}")

            return {
                "success": True,
                "chart_html": fig.to_html(include_plotlyjs='cdn'),
                "chart_json": json.dumps(fig, cls=PlotlyJSONEncoder)
            }

        except Exception as e:
            logger.error(f"Error creating pie chart: {str(e)}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def _create_scatter_chart(df: pd.DataFrame, x_column: Optional[str],
                             y_column: Optional[str], group_by: Optional[str],
                             title: Optional[str]) -> Dict[str, Any]:
        """创建散点图"""
        try:
            # 选择数值列
            numeric_cols = df.select_dtypes(include=['number']).columns

            if not x_column:
                x_column = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]
            if not y_column:
                y_column = numeric_cols[1] if len(numeric_cols) > 1 else df.columns[1]

            fig = px.scatter(df, x=x_column, y=y_column, color=group_by,
                           title=title or f"{y_column} vs {x_column}",
                           hover_data=[col for col in df.columns if col not in [x_column, y_column]])

            return {
                "success": True,
                "chart_html": fig.to_html(include_plotlyjs='cdn'),
                "chart_json": json.dumps(fig, cls=PlotlyJSONEncoder)
            }

        except Exception as e:
            logger.error(f"Error creating scatter chart: {str(e)}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def _create_histogram(df: pd.DataFrame, x_column: Optional[str],
                         title: Optional[str]) -> Dict[str, Any]:
        """创建直方图"""
        try:
            if not x_column:
                # 选择第一个数值列
                numeric_cols = df.select_dtypes(include=['number']).columns
                x_column = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]

            fig = px.histogram(df, x=x_column, title=title or f"Distribution of {x_column}")
            fig.update_layout(bargap=0.1)

            return {
                "success": True,
                "chart_html": fig.to_html(include_plotlyjs='cdn'),
                "chart_json": json.dumps(fig, cls=PlotlyJSONEncoder)
            }

        except Exception as e:
            logger.error(f"Error creating histogram: {str(e)}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def _create_box_chart(df: pd.DataFrame, x_column: Optional[str],
                         y_column: Optional[str], group_by: Optional[str],
                         title: Optional[str]) -> Dict[str, Any]:
        """创建箱线图"""
        try:
            if not x_column:
                numeric_cols = df.select_dtypes(include=['number']).columns
                x_column = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]

            fig = px.box(df, y=x_column, x=group_by, color=group_by,
                        title=title or f"Box plot of {x_column}")

            return {
                "success": True,
                "chart_html": fig.to_html(include_plotlyjs='cdn'),
                "chart_json": json.dumps(fig, cls=PlotlyJSONEncoder)
            }

        except Exception as e:
            logger.error(f"Error creating box chart: {str(e)}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def _create_heatmap(df: pd.DataFrame, title: Optional[str]) -> Dict[str, Any]:
        """创建热力图"""
        try:
            # 只选择数值列
            numeric_df = df.select_dtypes(include=['number'])

            if numeric_df.empty:
                return {"success": False, "error": "No numeric columns for heatmap"}

            # 计算相关性矩阵
            corr_matrix = numeric_df.corr()

            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                          title=title or "Correlation Heatmap")

            return {
                "success": True,
                "chart_html": fig.to_html(include_plotlyjs='cdn'),
                "chart_json": json.dumps(fig, cls=PlotlyJSONEncoder)
            }

        except Exception as e:
            logger.error(f"Error creating heatmap: {str(e)}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def create_summary_stats(data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        创建数据摘要统计可视化

        Args:
            data: 数据列表

        Returns:
            摘要统计HTML
        """
        try:
            df = pd.DataFrame(data)

            if df.empty:
                return {"success": False, "error": "No data to analyze"}

            # 创建摘要报告
            summary_html = f"""
            <div style="padding: 20px; font-family: Arial, sans-serif;">
                <h2>数据摘要报告</h2>
                <p><strong>数据行数:</strong> {len(df)}</p>
                <p><strong>数据列数:</strong> {len(df.columns)}</p>

                <h3>列信息</h3>
                <table border="1" style="border-collapse: collapse; width: 100%;">
                    <tr><th>列名</th><th>数据类型</th><th>非空值</th><th>唯一值</th></tr>
            """

            for col in df.columns:
                non_null = df[col].count()
                unique = df[col].nunique()
                dtype = str(df[col].dtype)
                summary_html += f"<tr><td>{col}</td><td>{dtype}</td><td>{non_null}</td><td>{unique}</td></tr>"

            summary_html += "</table>"

            # 数值列统计
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                summary_html += "<h3>数值列统计</h3><table border='1' style='border-collapse: collapse; width: 100%;'><tr><th>列名</th><th>平均值</th><th>标准差</th><th>最小值</th><th>最大值</th></tr>"

                for col in numeric_cols:
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    min_val = df[col].min()
                    max_val = df[col].max()
                    summary_html += f"<tr><td>{col}</td><td>{mean_val:.2f}</td><td>{std_val:.2f}</td><td>{min_val:.2f}</td><td>{max_val:.2f}</td></tr>"

                summary_html += "</table>"

            summary_html += "</div>"

            return {
                "success": True,
                "summary_html": summary_html
            }

        except Exception as e:
            logger.error(f"Error creating summary stats: {str(e)}")
            return {"success": False, "error": str(e)}