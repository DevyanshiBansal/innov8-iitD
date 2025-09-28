"""
Startup script for the FastAPI backend server
"""
from server import app
import uvicorn

if __name__ == "__main__":
    print("Starting Code Editor Backend Server...")
    print("Server will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)