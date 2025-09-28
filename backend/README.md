# Code Editor Backend Server

A FastAPI-based backend server for the React code editor frontend, providing code execution, time complexity analysis, and error handling.

## Features

- **Code Execution**: Execute Python, C++, and C code
- **Time Complexity Analysis**: Automatically analyze performance with different input sizes (10, 100, 500, 1000, 10000)
- **Memory Monitoring**: Track peak memory usage during execution
- **Error Analysis**: Intelligent error detection and explanation
- **Frontend Integration**: Seamless CORS setup for React frontend on localhost:3000

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Server**:
   ```bash
   python server.py
   ```

3. **Server will be available at**: http://localhost:8000

## API Endpoints

### Main Endpoints
- `POST /api/execute` - Execute code with analysis
- `GET /api/health` - Health check
- `GET /api/status` - Server status and features
- `GET /api/supported-languages` - Get supported languages with examples
- `GET /api/default-test-sizes` - Get default test sizes for analysis

### Frontend Integration
The backend is configured to work with the React frontend running on `http://localhost:3000`.

### Execute Code Endpoint

**Request Format**:
```json
{
  "code": "def main(n):\n    return sum(range(n))\n\nresult = main(10)\nprint(f'Result: {result}')",
  "language": "python",
  "test_sizes": [10, 100, 500, 1000, 10000]
}
```

**Response Format**:
```json
{
  "success": true,
  "message": "Code executed successfully",
  "data": {
    "execution_output": {
      "output": "Result: 45",
      "error": "",
      "exit_code": 0,
      "execution_time": 0.1,
      "peak_memory": 1024,
      "peak_memory_formatted": "1.00 KB"
    },
    "time_complexity": {
      "time_measurements": {"10": 0.001, "100": 0.002, "500": 0.01, "1000": 0.02, "10000": 0.2},
      "memory_measurements": {"10": 1024, "100": 1024, "500": 1024, "1000": 1024, "10000": 1024}
    },
    "error_analysis": "No Errors"
  },
  "frontend_display": {
    "output": "Result: 45",
    "n_vs_time": {"10": 0.001, "100": 0.002, "500": 0.01, "1000": 0.02, "10000": 0.2},
    "n_vs_memory": {"10": 1024, "100": 1024, "500": 1024, "1000": 1024, "10000": 1024},
    "error_analysis": "No Errors",
    "execution_summary": {
      "exit_code": 0,
      "execution_time": 0.1,
      "peak_memory": "1.00 KB",
      "success": true
    }
  }
}
```

## Supported Languages

- **Python** (.py)
- **C++** (.cpp)
- **C** (.c)

## Chat Panel Integration

The backend provides exactly what the frontend chat panel needs:

1. **n vs time dictionary**: Available in `frontend_display.n_vs_time`
2. **n vs peak memory dictionary**: Available in `frontend_display.n_vs_memory`
3. **Error analysis**: Available in `frontend_display.error_analysis`

## Dependencies

- FastAPI 0.104.1
- uvicorn 0.24.0
- pydantic 2.5.0
- python-multipart 0.0.6
- psutil 5.9.6

## Development

The server uses the existing `executor_local.py` for code execution and analysis, providing a clean API interface for the frontend React application.