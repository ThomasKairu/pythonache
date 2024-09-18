"""
This module serves as the entry point for running the FastAPI application
using Uvicorn server.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
