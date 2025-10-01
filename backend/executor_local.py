import os
import subprocess
import time
import psutil
import tempfile
import json
from typing import Dict, Any, List, Tuple
import threading
import sys

# Import resource module conditionally (Unix only)
try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:
    RESOURCE_AVAILABLE = False

def execute_with_complexity_analysis(code: str, language: str, stdin: str = "") -> Dict[str, Any]:
    """
    Execute code and analyze time/space complexity with multiple input sizes
    """
    try:
        # Create temporary file for code
        with tempfile.NamedTemporaryFile(mode='w', suffix=get_file_extension(language), delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        # Prepare execution command
        cmd = get_execution_command(language, temp_file)
        
        if not cmd:
            return create_error_result(f"Unsupported language: {language}")
        
        # Execute with different input sizes for complexity analysis
        time_measurements = {}
        memory_measurements = {}
        
        # Test with different input sizes
        test_sizes = [10, 100, 500, 1000, 5000]
        
        for size in test_sizes:
            try:
                # Generate input based on size
                test_input = generate_test_input(size, stdin)
                
                # Measure execution
                start_time = time.perf_counter()
                start_memory = get_memory_usage()
                
                result = subprocess.run(
                    cmd,
                    input=test_input,
                    capture_output=True,
                    text=True,
                    timeout=30,  # 30 second timeout
                    check=False
                )
                
                end_time = time.perf_counter()
                end_memory = get_memory_usage()
                
                execution_time = end_time - start_time
                memory_used = max(end_memory - start_memory, 0)
                
                time_measurements[size] = execution_time
                memory_measurements[size] = memory_used
                
                # Store the actual output from the first successful run
                if size == test_sizes[0]:
                    # Capture both stdout and stderr
                    actual_output = result.stdout.strip() if result.stdout else ""
                    actual_error = result.stderr.strip() if result.stderr else ""
                    
                    # If there's stdout, use it; otherwise use stderr
                    combined_output = actual_output
                    if not combined_output and actual_error:
                        combined_output = actual_error
                    elif combined_output and actual_error:
                        # Both stdout and stderr exist, combine them
                        combined_output = f"{actual_output}\n--- STDERR ---\n{actual_error}"
                    
                    exit_code = result.returncode
                
            except subprocess.TimeoutExpired:
                time_measurements[size] = float('inf')
                memory_measurements[size] = 0
                break
            except Exception as e:
                time_measurements[size] = float('inf')
                memory_measurements[size] = 0
                continue
        
        # Clean up temporary file
        try:
            os.unlink(temp_file)
        except:
            pass
        
        # Analyze complexity patterns
        time_complexity_analysis = analyze_time_complexity(time_measurements)
        space_complexity_analysis = analyze_space_complexity(memory_measurements)
        
        # Format results
        return {
            "success": exit_code == 0,
            "output": combined_output if 'combined_output' in locals() else "",
            "execution_time": time_measurements.get(test_sizes[0], 0),
            "peak_memory": format_memory(memory_measurements.get(test_sizes[0], 0)),
            "time_complexity": {
                "measurements": time_measurements,
                "analysis": time_complexity_analysis
            },
            "space_complexity": {
                "measurements": memory_measurements,
                "analysis": space_complexity_analysis
            },
            "error_analysis": analyze_errors(actual_error if 'actual_error' in locals() and exit_code != 0 else "")
        }
        
    except Exception as e:
        return create_error_result(f"Execution failed: {str(e)}")

def get_file_extension(language: str) -> str:
    """Get file extension for the given language"""
    extensions = {
        "python": ".py",
        "cpp": ".cpp",
        "c": ".c",
        "java": ".java",
        "javascript": ".js",
        "typescript": ".ts"
    }
    return extensions.get(language.lower(), ".txt")

def get_execution_command(language: str, file_path: str) -> List[str]:
    """Get execution command for the given language"""
    commands = {
        "python": ["python", file_path],
        "cpp": None,  # Needs compilation
        "c": None,    # Needs compilation
        "java": None, # Needs compilation
        "javascript": ["node", file_path],
        "typescript": ["ts-node", file_path]
    }
    
    lang = language.lower()
    
    if lang in ["cpp", "c"]:
        # Compile first
        executable = file_path.replace(get_file_extension(lang), ".exe")
        compiler = "g++" if lang == "cpp" else "gcc"
        
        try:
            compile_result = subprocess.run(
                [compiler, file_path, "-o", executable],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if compile_result.returncode == 0:
                return [executable]
            else:
                return None
        except:
            return None
    
    elif lang == "java":
        # Compile Java
        try:
            class_name = os.path.basename(file_path).replace(".java", "")
            compile_result = subprocess.run(
                ["javac", file_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if compile_result.returncode == 0:
                return ["java", "-cp", os.path.dirname(file_path), class_name]
            else:
                return None
        except:
            return None
    
    return commands.get(lang)

def generate_test_input(size: int, base_input: str) -> str:
    """Generate test input based on size parameter"""
    if base_input.strip():
        # If user provided input, use it as-is for smaller sizes
        if size <= 100:
            return base_input
        else:
            # For larger sizes, try to scale the input
            lines = base_input.strip().split('\n')
            if lines and lines[0].isdigit():
                # If first line is a number, scale it
                return str(size) + '\n' + '\n'.join(lines[1:])
            else:
                return base_input
    else:
        # Generate default test input
        return f"{size}\n" + " ".join(map(str, range(1, min(size + 1, 100))))

def get_memory_usage() -> int:
    """Get current memory usage in bytes"""
    try:
        process = psutil.Process()
        return process.memory_info().rss
    except:
        return 0

def analyze_time_complexity(measurements: Dict[int, float]) -> str:
    """Analyze time complexity based on measurements"""
    if not measurements or len(measurements) < 2:
        return "Insufficient data for analysis"
    
    # Remove infinite values
    valid_measurements = {k: v for k, v in measurements.items() if v != float('inf')}
    
    if len(valid_measurements) < 2:
        return "Execution timeout - possible infinite loop or high complexity"
    
    # Sort by input size
    sorted_data = sorted(valid_measurements.items())
    
    # Calculate ratios between consecutive measurements
    ratios = []
    for i in range(1, len(sorted_data)):
        prev_size, prev_time = sorted_data[i-1]
        curr_size, curr_time = sorted_data[i]
        
        if prev_time > 0:
            size_ratio = curr_size / prev_size
            time_ratio = curr_time / prev_time
            ratios.append((size_ratio, time_ratio))
    
    if not ratios:
        return "O(1) - Constant time"
    
    # Analyze patterns
    avg_time_ratio = sum(r[1] for r in ratios) / len(ratios)
    avg_size_ratio = sum(r[0] for r in ratios) / len(ratios)
    
    if avg_time_ratio <= avg_size_ratio * 1.2:  # Allow some variance
        return "O(n) - Linear time"
    elif avg_time_ratio <= avg_size_ratio ** 1.5:
        return "O(n log n) - Linearithmic time"
    elif avg_time_ratio <= avg_size_ratio ** 2.2:
        return "O(n²) - Quadratic time"
    elif avg_time_ratio > avg_size_ratio ** 2.5:
        return "O(n³) or higher - High complexity"
    else:
        return "O(1) - Constant time"

def analyze_space_complexity(measurements: Dict[int, int]) -> str:
    """Analyze space complexity based on memory measurements"""
    if not measurements or len(measurements) < 2:
        return "Insufficient data for analysis"
    
    # Remove zero values
    valid_measurements = {k: v for k, v in measurements.items() if v > 0}
    
    if len(valid_measurements) < 2:
        return "O(1) - Constant space"
    
    # Sort by input size
    sorted_data = sorted(valid_measurements.items())
    
    # Check if memory usage grows with input size
    memory_growth = []
    for i in range(1, len(sorted_data)):
        prev_size, prev_memory = sorted_data[i-1]
        curr_size, curr_memory = sorted_data[i]
        
        size_growth = curr_size / prev_size if prev_size > 0 else 1
        memory_ratio = curr_memory / prev_memory if prev_memory > 0 else 1
        memory_growth.append(memory_ratio / size_growth)
    
    if not memory_growth:
        return "O(1) - Constant space"
    
    avg_growth = sum(memory_growth) / len(memory_growth)
    
    if avg_growth <= 1.2:
        return "O(1) - Constant space"
    elif avg_growth <= 2:
        return "O(n) - Linear space"
    else:
        return "O(n²) or higher - High space complexity"

def format_memory(memory_bytes: int) -> str:
    """Format memory usage in human readable format"""
    if memory_bytes < 1024:
        return f"{memory_bytes} B"
    elif memory_bytes < 1024 * 1024:
        return f"{memory_bytes / 1024:.1f} KB"
    elif memory_bytes < 1024 * 1024 * 1024:
        return f"{memory_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{memory_bytes / (1024 * 1024 * 1024):.1f} GB"

def analyze_errors(error_output: str) -> str:
    """Analyze error output and provide helpful feedback"""
    if not error_output:
        return "No errors detected"
    
    error_lower = error_output.lower()
    
    if "syntaxerror" in error_lower:
        return "Syntax Error: Check for missing parentheses, brackets, or incorrect indentation"
    elif "nameerror" in error_lower:
        return "Name Error: Variable or function not defined before use"
    elif "indexerror" in error_lower:
        return "Index Error: Array/list index out of bounds"
    elif "typeerror" in error_lower:
        return "Type Error: Incorrect data type operation"
    elif "valueerror" in error_lower:
        return "Value Error: Invalid value for the operation"
    elif "zerodivisionerror" in error_lower:
        return "Zero Division Error: Division by zero"
    elif "timeout" in error_lower:
        return "Timeout Error: Code took too long to execute - possible infinite loop"
    elif "compilation" in error_lower:
        return "Compilation Error: Code failed to compile - check syntax"
    else:
        return f"Runtime Error: {error_output[:200]}..."

def create_error_result(error_message: str) -> Dict[str, Any]:
    """Create a standard error result"""
    return {
        "success": False,
        "output": "",
        "execution_time": 0.0,
        "peak_memory": "0 B",
        "time_complexity": {
            "measurements": {},
            "analysis": "Error occurred during execution"
        },
        "space_complexity": {
            "measurements": {},
            "analysis": "Error occurred during execution"
        },
        "error_analysis": error_message
    }

if __name__ == "__main__":
    # Test the executor
    test_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

n = int(input())
print(fibonacci(n))
    """
    
    result = execute_with_complexity_analysis(test_code, "python", "10")
    print(json.dumps(result, indent=2))