import requests
import json

# Test code with print statements
test_code = '''
print("Starting calculation...")
x = 5
y = 10
result = x + y
print(f"The result of {x} + {y} is {result}")
print("Calculation completed!")
'''

url = "http://localhost:8000/api/execute"
payload = {
    "code": test_code,
    "language": "python",
    "stdin": ""
}

print("🧪 Testing print statement capture...")
try:
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success: {result['success']}")
        print(f"📤 Output captured: '{result['output']}'")
        print(f"⏱️ Execution time: {result['execution_time']}s")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
except Exception as e:
    print(f"❌ Request failed: {e}")