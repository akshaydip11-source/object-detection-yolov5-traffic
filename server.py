"""Run the SafeCityAI FastAPI application.

Usage:
    python server.py
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=False)
