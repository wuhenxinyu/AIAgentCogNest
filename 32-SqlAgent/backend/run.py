
"""
运行后端服务器的启动脚本
"""

import uvicorn
from app.config import settings

if __name__ == "__main__":
    print(f"""
    ===================================
    SQL Agent Backend Server
    ===================================
    Starting server on {settings.host}:{settings.port}
    Debug mode: {settings.debug}
    ===================================
    """)

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )