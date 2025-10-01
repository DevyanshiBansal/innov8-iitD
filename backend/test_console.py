import requests
import json

# Test the execute endpoint
url = "http://localhost:8000/api/execute"
payload = {
    "code": "print('Hello from console test!')\nprint('Testing console output!')",
    "language": "python", 
    "stdin": ""
}

print("🧪 Testing console output functionality...")
print("Sending request to /api/execute endpoint...")

try:
    response = requests.post(url, json=payload)
    print(f"✅ Response Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"📤 API Response Output: {result['output']}")
        print(f"⏱️ Execution Time: {result['execution_time']}s")
        print(f"✅ Success: {result['success']}")
    else:
        print(f"❌ Error: {response.text}")
        
except Exception as e:
    print(f"❌ Request failed: {e}")