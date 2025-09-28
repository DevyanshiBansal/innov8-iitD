"""
Solution Format Code Executor

This module handles LeetCode-style code execution where users must implement
a solution() function. It provides boilerplates and enforces the solution() naming convention.
"""

import os
import tempfile
import subprocess
import sys
import time
import json
import re
from typing import Dict, Any, Optional, List
import threading

# Import memory monitoring from original executor
from executor_local import MemoryMonitor, format_memory_usage

class SolutionExecutor:
    """
    Executes code in LeetCode format where users must implement solution() function
    """
    
    def __init__(self):
        self.supported_languages = {
            'python': {
                'extension': '.py',
                'command': [sys.executable],
                'boilerplate': self._get_python_boilerplate()
            },
            'javascript': {
                'extension': '.js',
                'command': ['node'],
                'boilerplate': self._get_javascript_boilerplate()
            },
            'java': {
                'extension': '.java',
                'command': ['java'],
                'compile_command': ['javac'],
                'boilerplate': self._get_java_boilerplate()
            },
            'cpp': {
                'extension': '.cpp',
                'command': None,  # Will be set after compilation
                'compile_command': ['g++', '-o'],
                'boilerplate': self._get_cpp_boilerplate()
            }
        }
    
    def get_language_boilerplate(self, language: str) -> str:
        """Get the boilerplate code for a specific language"""
        if language not in self.supported_languages:
            return f"# Language {language} not supported"
        
        return self.supported_languages[language]['boilerplate']
    
    def _get_python_boilerplate(self) -> str:
        return '''def solution():
    """
    Implement your solution here.
    
    Example:
    def solution(nums, target):
        # Your code here
        return result
    """
    pass

# Test your solution
if __name__ == "__main__":
    # Example usage:
    # result = solution([2, 7, 11, 15], 9)
    # print(result)
    
    # You can also read from input:
    # import sys
    # input_data = input().strip()
    # result = solution(input_data)
    # print(result)
    
    print("Please implement the solution() function")'''
    
    def _get_javascript_boilerplate(self) -> str:
        return '''function solution() {
    /**
     * Implement your solution here.
     * 
     * Example:
     * function solution(nums, target) {
     *     // Your code here
     *     return result;
     * }
     */
    return null;
}

// Test your solution
if (require.main === module) {
    // Example usage:
    // const result = solution([2, 7, 11, 15], 9);
    // console.log(result);
    
    // You can also read from input:
    // const readline = require('readline');
    // const rl = readline.createInterface({
    //     input: process.stdin,
    //     output: process.stdout
    // });
    // rl.question('', (input) => {
    //     const result = solution(input.trim());
    //     console.log(result);
    //     rl.close();
    // });
    
    console.log("Please implement the solution() function");
}'''
    
    def _get_java_boilerplate(self) -> str:
        return '''import java.util.*;
import java.io.*;

public class Solution {
    public static Object solution() {
        /**
         * Implement your solution here.
         * 
         * Example:
         * public static int[] solution(int[] nums, int target) {
         *     // Your code here
         *     return result;
         * }
         */
        return null;
    }
    
    public static void main(String[] args) {
        // Example usage:
        // int[] result = solution(new int[]{2, 7, 11, 15}, 9);
        // System.out.println(Arrays.toString(result));
        
        // You can also read from input:
        // Scanner scanner = new Scanner(System.in);
        // String input = scanner.nextLine();
        // Object result = solution(input);
        // System.out.println(result);
        
        System.out.println("Please implement the solution() function");
    }
}'''
    
    def _get_cpp_boilerplate(self) -> str:
        return '''#include <iostream>
#include <vector>
#include <string>
using namespace std;

class Solution {
public:
    /**
     * Implement your solution here.
     * 
     * Example:
     * vector<int> solution(vector<int>& nums, int target) {
     *     // Your code here
     *     return result;
     * }
     */
    auto solution() {
        return nullptr;
    }
};

int main() {
    Solution sol;
    
    // Example usage:
    // vector<int> nums = {2, 7, 11, 15};
    // int target = 9;
    // auto result = sol.solution(nums, target);
    
    // You can also read from input:
    // string input;
    // getline(cin, input);
    // auto result = sol.solution(input);
    // cout << result << endl;
    
    cout << "Please implement the solution() function" << endl;
    return 0;
}'''

    def execute_solution_code(
        self,
        code: str,
        language: str,
        test_input: Optional[str] = None,
        custom_test_sizes: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Execute code in solution format with optional test input
        
        Args:
            code: The user's code containing solution() function
            language: Programming language
            test_input: Optional input to pass to the solution
            custom_test_sizes: Sizes for complexity analysis
            
        Returns:
            Execution results with output, errors, and performance data
        """
        
        if language not in self.supported_languages:
            return {
                "output": "",
                "error": f"Language {language} not supported",
                "exit_code": -1,
                "execution_time": 0,
                "peak_memory": 0,
                "peak_memory_formatted": "0 B",
                "solution_found": False
            }
        
        # Validate that code contains solution() function
        if not self._validate_solution_function(code, language):
            return {
                "output": "",
                "error": f"Code must contain a 'solution()' function. Use the provided boilerplate.",
                "exit_code": -1,
                "execution_time": 0,
                "peak_memory": 0,
                "peak_memory_formatted": "0 B",
                "solution_found": False
            }
        
        # Create temporary file with the code
        try:
            temp_file = self._create_temp_file(code, language, test_input)
            
            # Execute the code
            result = self._execute_temp_file(temp_file, language)
            
            # Clean up temporary file
            try:
                os.unlink(temp_file)
                # Also clean up compiled files for Java/C++
                if language == 'java':
                    class_file = temp_file.replace('.java', '.class')
                    if os.path.exists(class_file):
                        os.unlink(class_file)
                elif language == 'cpp':
                    exe_file = temp_file.replace('.cpp', '.exe')
                    if os.path.exists(exe_file):
                        os.unlink(exe_file)
            except:
                pass  # Ignore cleanup errors
            
            result["solution_found"] = True
            return result
            
        except Exception as e:
            return {
                "output": "",
                "error": f"Execution failed: {str(e)}",
                "exit_code": -1,
                "execution_time": 0,
                "peak_memory": 0,
                "peak_memory_formatted": "0 B",
                "solution_found": False
            }
    
    def _validate_solution_function(self, code: str, language: str) -> bool:
        """Check if code contains a solution() function"""
        if language == 'python':
            return 'def solution(' in code
        elif language == 'javascript':
            return 'function solution(' in code or 'const solution =' in code or 'let solution =' in code
        elif language == 'java':
            return 'solution(' in code and 'public' in code
        elif language == 'cpp':
            return 'solution(' in code
        return False
    
    def _create_temp_file(self, code: str, language: str, test_input: Optional[str] = None) -> str:
        """Create temporary file with user code and optional test input wrapper"""
        lang_info = self.supported_languages[language]
        extension = lang_info['extension']
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix=extension, delete=False) as temp_file:
            if test_input:
                # Wrap the code with test input handling
                wrapped_code = self._wrap_code_with_input(code, language, test_input)
                temp_file.write(wrapped_code)
            else:
                temp_file.write(code)
            
            return temp_file.name
    
    def _wrap_code_with_input(self, code: str, language: str, test_input: str) -> str:
        """Wrap user code to handle test input"""
        if language == 'python':
            # For Python, we'll add input handling
            wrapper = f'''
import sys
import json

# User's code
{code}

# Test input handling
if __name__ == "__main__":
    try:
        test_input = """{test_input}"""
        if test_input.strip():
            # Try to parse as JSON first, then as string
            try:
                parsed_input = json.loads(test_input)
                if isinstance(parsed_input, list):
                    result = solution(*parsed_input)
                else:
                    result = solution(parsed_input)
            except:
                # Treat as string input
                result = solution(test_input.strip())
        else:
            result = solution()
        
        print(result)
    except Exception as e:
        print(f"Error: {{e}}")
'''
            return wrapper
        
        elif language == 'javascript':
            wrapper = f'''
{code}

// Test input handling
if (require.main === module) {{
    try {{
        const testInput = `{test_input}`;
        let result;
        
        if (testInput.trim()) {{
            try {{
                const parsedInput = JSON.parse(testInput);
                if (Array.isArray(parsedInput)) {{
                    result = solution(...parsedInput);
                }} else {{
                    result = solution(parsedInput);
                }}
            }} catch {{
                result = solution(testInput.trim());
            }}
        }} else {{
            result = solution();
        }}
        
        console.log(result);
    }} catch (e) {{
        console.log(`Error: ${{e.message}}`);
    }}
}}
'''
            return wrapper
        
        # For Java and C++, return original code for now (more complex to wrap)
        return code
    
    def _execute_temp_file(self, temp_file: str, language: str) -> Dict[str, Any]:
        """Execute the temporary file and return results"""
        lang_info = self.supported_languages[language]
        
        # Handle compilation for compiled languages
        if language == 'java':
            return self._execute_java(temp_file)
        elif language == 'cpp':
            return self._execute_cpp(temp_file)
        else:
            # Direct execution for interpreted languages
            command = lang_info['command'] + [temp_file]
            return self._run_command(command)
    
    def _execute_java(self, java_file: str) -> Dict[str, Any]:
        """Compile and execute Java file"""
        try:
            # Compile
            compile_result = subprocess.run(
                ['javac', java_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if compile_result.returncode != 0:
                return {
                    "output": "",
                    "error": f"Compilation failed:\n{compile_result.stderr}",
                    "exit_code": compile_result.returncode,
                    "execution_time": 0,
                    "peak_memory": 0,
                    "peak_memory_formatted": "0 B"
                }
            
            # Execute
            class_name = os.path.basename(java_file).replace('.java', '')
            class_file_dir = os.path.dirname(java_file)
            
            command = ['java', '-cp', class_file_dir, class_name]
            return self._run_command(command)
            
        except Exception as e:
            return {
                "output": "",
                "error": f"Java execution error: {str(e)}",
                "exit_code": -1,
                "execution_time": 0,
                "peak_memory": 0,
                "peak_memory_formatted": "0 B"
            }
    
    def _execute_cpp(self, cpp_file: str) -> Dict[str, Any]:
        """Compile and execute C++ file"""
        try:
            exe_file = cpp_file.replace('.cpp', '.exe')
            
            # Compile
            compile_result = subprocess.run(
                ['g++', cpp_file, '-o', exe_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if compile_result.returncode != 0:
                return {
                    "output": "",
                    "error": f"Compilation failed:\n{compile_result.stderr}",
                    "exit_code": compile_result.returncode,
                    "execution_time": 0,
                    "peak_memory": 0,
                    "peak_memory_formatted": "0 B"
                }
            
            # Execute
            return self._run_command([exe_file])
            
        except Exception as e:
            return {
                "output": "",
                "error": f"C++ execution error: {str(e)}",
                "exit_code": -1,
                "execution_time": 0,
                "peak_memory": 0,
                "peak_memory_formatted": "0 B"
            }
    
    def _run_command(self, command: List[str]) -> Dict[str, Any]:
        """Run command and monitor execution"""
        memory_monitor = MemoryMonitor()
        start_time = time.time()
        
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            memory_monitor.start_monitoring(process)
            stdout, stderr = process.communicate(timeout=10)  # 10 second timeout
            memory_monitor.stop_monitoring()
            
            execution_time = time.time() - start_time
            
            return {
                "output": stdout.strip(),
                "error": stderr.strip() if stderr else "",
                "exit_code": process.returncode,
                "execution_time": execution_time,
                "peak_memory": memory_monitor.peak_memory,
                "peak_memory_formatted": format_memory_usage(memory_monitor.peak_memory)
            }
            
        except subprocess.TimeoutExpired:
            process.kill()
            memory_monitor.stop_monitoring()
            return {
                "output": "",
                "error": "Execution timed out (10 seconds)",
                "exit_code": -1,
                "execution_time": 10.0,
                "peak_memory": memory_monitor.peak_memory,
                "peak_memory_formatted": format_memory_usage(memory_monitor.peak_memory)
            }
        except Exception as e:
            memory_monitor.stop_monitoring()
            return {
                "output": "",
                "error": f"Execution error: {str(e)}",
                "exit_code": -1,
                "execution_time": time.time() - start_time,
                "peak_memory": 0,
                "peak_memory_formatted": "0 B"
            }

# Global instance
_solution_executor = SolutionExecutor()

def get_solution_executor() -> SolutionExecutor:
    """Get the global solution executor instance"""
    return _solution_executor

def execute_solution_code(
    code: str,
    language: str,
    test_input: Optional[str] = None,
    custom_test_sizes: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Convenience function for executing solution code"""
    executor = get_solution_executor()
    return executor.execute_solution_code(code, language, test_input, custom_test_sizes)

def get_language_boilerplate(language: str) -> str:
    """Get boilerplate code for a language"""
    executor = get_solution_executor()
    return executor.get_language_boilerplate(language)

def get_supported_languages() -> List[str]:
    """Get list of supported languages"""
    executor = get_solution_executor()
    return list(executor.supported_languages.keys())


# Test the solution executor
if __name__ == "__main__":
    # Test Python boilerplate
    python_code = '''def solution(nums, target):
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
'''
    
    test_input = '[2, 7, 11, 15], 9'
    result = execute_solution_code(python_code, 'python', test_input)
    
    print("Test Result:")
    print(json.dumps(result, indent=2))