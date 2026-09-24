"""Run the SafeCityAI FastAPI application.

Usage:
    python server.py
"""

import os
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
        access_log=False,
    )
