from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class FileType(str, Enum):
    CSV = "csv"
    EXCEL = "excel"
    XLSX = "xlsx"
    XLS = "xls"


class FileUploadResponse(BaseModel):
    success: bool
    file_id: Optional[str] = None
    message: str
    headers: Optional[List[str]] = None
    column_info: Optional[List[Dict[str, Any]]] = None
    total_columns: Optional[int] = None
    estimated_rows: Optional[int] = None


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    file_id: Optional[str] = Field(None, description="File ID if querying uploaded file")
    table_name: Optional[str] = Field(None, description="Table name if querying database table")
    columns: Optional[List[str]] = Field(None, description="Specific columns to query")
    limit: Optional[int] = Field(None, description="Maximum number of rows to return (deprecated, use natural language in query)")


class QueryResponse(BaseModel):
    success: bool
    data: Optional[List[Dict[str, Any]]] = None
    answer: Optional[str] = None
    sql: Optional[str] = None
    reasoning: Optional[List[str]] = None
    total_rows: Optional[int] = None
    returned_rows: Optional[int] = None
    columns: Optional[List[str]] = None
    error: Optional[str] = None
    visualization: Optional[str] = None


class ChartType(str, Enum):
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    HEATMAP = "heatmap"


class VisualizationRequest(BaseModel):
    file_id: Optional[str] = None
    chart_type: ChartType
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    group_by: Optional[str] = None
    title: Optional[str] = None
    limit: int = Field(100, description="Maximum data points for visualization")


class VisualizationResponse(BaseModel):
    success: bool
    chart_url: Optional[str] = None
    chart_html: Optional[str] = None
    error: Optional[str] = None


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = None


class ChatSession(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    file_id: Optional[str] = None
    created_at: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Existing session ID")
    file_id: Optional[str] = Field(None, description="File ID to analyze")


class ChatResponse(BaseModel):
    success: bool
    message: str
    session_id: str
    data: Optional[List[Dict[str, Any]]] = None
    visualization: Optional[str] = None
    error: Optional[str] = None