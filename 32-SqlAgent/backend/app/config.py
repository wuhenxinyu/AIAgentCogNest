from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # API Keys
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # LangSmith Configuration
    langchain_tracing_v2: bool = False
    langchain_api_key: Optional[str] = None
    langchain_project: str = "sql-agent-project"

    # Database Configuration
    database_url: str = "sqlite:///./data/sql_agent.db"

    # FastAPI Configuration
    host: str = "0.0.0.0"
    port: int = 8001
    debug: bool = True

    # File Upload Configuration
    max_file_size: str = "100MB"
    upload_dir: str = "./data/uploads"

    # Visualization Configuration
    vis_output_dir: str = "./data/visualizations"

    # Model Configuration
    default_model: str = "gpt-3.5-turbo"
    temperature: float = 0.0

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


# Initialize settings
settings = Settings()

# Configure LangSmith if enabled
if settings.langchain_tracing_v2 and settings.langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2)
    os.environ["LANGCHAIN_API_KEY"] = str(settings.langchain_api_key)
    os.environ["LANGCHAIN_PROJECT"] = str(settings.langchain_project)