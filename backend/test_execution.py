"""
Test script to validate the integrated execution system
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

# Import server functions
from server import execute_code_with_subprocess

def test_execution():
    """Test the integrated execution system"""
    
    # Test code with print statements
    test_code = """
def main():
    print("Hello from subprocess!")
    print("Testing OUTPUT display")
    result = 2 + 2
    print(f"Result: {result}")
    return result

if __name__ == "__main__":
    main()
"""
    
    print("Testing integrated execution system...")
    print("=" * 50)
    
    # Execute the test code
    result = execute_code_with_subprocess(test_code, "python")
    
    print("Execution Result:")
    print(f"Success: {result['success']}")
    print(f"Exit Code: {result['exit_code']}")
    print(f"Execution Time: {result['execution_time']}s")
    print(f"Peak Memory: {result['peak_memory_formatted']}")
    
    print("\nOUTPUT:")
    if result['output']:
        output_lines = result['output'].strip().split('\n')
        for line in output_lines:
            if line.strip():
                print(line)
    
    if result['error']:
        print(f"\nERROR: {result['error']}")
    
    print("Next Level: Medium")
    print("=" * 50)

if __name__ == "__main__":
    test_execution()