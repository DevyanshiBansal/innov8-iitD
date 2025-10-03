importclass SimpleChatbot:
    """
    Simple Gemini-based chatbot that can answer questions about code, algorithms,
    and programming concepts.
    """
    
    def __init__(self, api_key: str = None):
        # Use provided Gemini API key - you'll need to provide this
        self.gemini_api_key = api_key or "_"  # Replace with your key
        
        # Initialize Gemini client
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')  # Use stable version
            print("✅ Gemini chatbot initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Gemini chatbot: {e}")
            self.gemini_model = Nonefrom typing import Dict, Any, List, Optional
import requests
import google.generativeai as genai

class SimpleChatbot:
    """
    Simple Gemini-based code analyzer that uses chain of thought and ReACT reasoning
    to analyze code complexity without requiring RAG or Langchain dependencies.
    """
    
    def __init__(self, api_key: str = None):
        # Use provided Gemini API key
        self.gemini_api_key = "_"
        
        # Initialize Gemini client
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            print("✅ Gemini client initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Gemini client: {e}")
            self.gemini_model = None
        
        # Complexity analysis rules for the LLM to use
        self.complexity_rules = """
        TIME COMPLEXITY ANALYSIS RULES:
        1. Single loop over n elements: O(n)
        2. Nested loops: O(n²), O(n³), etc. (multiply complexities)
        3. Binary search, divide & conquer: O(log n)
        4. Recursive algorithms: Calculate based on recurrence relation
        5. Hash table operations (average case): O(1)
        6. Sorting algorithms: O(n log n) for efficient ones, O(n²) for bubble/insertion
        7. Tree traversal: O(n) where n is number of nodes
        8. Graph algorithms: O(V + E) for DFS/BFS where V=vertices, E=edges
        9. Dynamic programming: Often O(n²) or O(n³) depending on subproblems
        
        SPACE COMPLEXITY ANALYSIS RULES:
        1. Constant extra space: O(1) - only using a few variables
        2. Recursive calls: O(h) where h is maximum recursion depth
        3. Arrays/lists of size n: O(n)
        4. 2D arrays: O(n²)
        5. Hash tables storing n elements: O(n)
        6. In-place algorithms: O(1) additional space (modifying input doesn't count)
        7. Memoization: O(number of unique subproblems)
        
        COMMON ALGORITHM PATTERNS:
        - Two pointers: Usually O(n) time, O(1) space
        - Sliding window: Usually O(n) time, O(1) space
        - Binary search: O(log n) time, O(1) space
        - DFS/BFS: O(V + E) time, O(V) space
        - Merge sort: O(n log n) time, O(n) space
        - Quick sort: O(n log n) average time, O(log n) space
        """
    
    def analyze_code(self, code: str, language: str, performance_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze code using Groq LLM with chain of thought reasoning
        
        Args:
            code: The source code to analyze
            language: Programming language (python, javascript, java, etc.)
            performance_data: Optional dict containing n_vs_time and n_vs_space data
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Create comprehensive analysis prompt
            prompt = self._create_analysis_prompt(code, language, performance_data or {})
            
            # Make API request to Gemini
            response = self._call_gemini_api(prompt)
            
            if response and 'choices' in response and len(response['choices']) > 0:
                analysis_text = response['choices'][0]['message']['content']
                
                # Parse the structured response
                parsed_analysis = self._parse_analysis_response(analysis_text)
                
                return {
                    "success": True,
                    "analysis": analysis_text,
                    "complexity": parsed_analysis.get("complexity", {"time": "O(n)", "space": "O(1)"}),
                    "suggestions": parsed_analysis.get("suggestions", []),
                    "reasoning": parsed_analysis.get("reasoning", ""),
                    "code_quality": parsed_analysis.get("code_quality", {}),
                    "performance_insights": parsed_analysis.get("performance_analysis", "")
                }
            else:
                return self._fallback_analysis(code, performance_data or {})
                
        except Exception as e:
            print(f"Error in Groq analysis: {e}")
            return self._fallback_analysis(code, performance_data or {})
    
    def _create_analysis_prompt(self, code: str, language: str, performance_data: Dict[str, Any]) -> str:
        """Create detailed analysis prompt with ReACT reasoning methodology"""
        
        performance_section = ""
        if performance_data:
            performance_section = f"""
PERFORMANCE DATA PROVIDED:
{json.dumps(performance_data, indent=2)}
"""
        
        return f"""
You are an expert algorithm and code complexity analyzer. Use ReACT (Reasoning, Acting, Observing) methodology with chain of thought reasoning to analyze this code thoroughly.

COMPLEXITY ANALYSIS RULES TO FOLLOW:
{self.complexity_rules}

CODE TO ANALYZE:
Language: {language}
```{language}
{code}
```

{performance_section}

ANALYSIS METHODOLOGY - Use ReACT Framework:

1. REASON: Think step by step about the code structure
   - What is the main algorithm or approach?
   - What data structures are used?
   - Are there loops, recursion, or other patterns?

2. ACT: Apply complexity analysis rules
   - Count loop nestings for time complexity
   - Identify space usage patterns
   - Look for known algorithm patterns

3. OBSERVE: Verify your analysis
   - Does the performance data (if provided) match your complexity analysis?
   - Are there any inconsistencies to address?
   - What optimizations could improve the algorithm?

PROVIDE YOUR ANALYSIS IN THIS EXACT JSON FORMAT:
{{
  "reasoning": "Step-by-step explanation of your thought process analyzing the code",
  "complexity": {{
    "time": "O(...)",
    "space": "O(...)",
    "time_explanation": "Detailed explanation of why this is the time complexity",
    "space_explanation": "Detailed explanation of why this is the space complexity"
  }},
  "code_quality": {{
    "score": 8,
    "issues": ["List any code issues found"],
    "strengths": ["List code strengths"],
    "readability": "Assessment of code readability"
  }},
  "suggestions": [
    "Specific optimization suggestion with explanation",
    "Alternative approach suggestion",
    "Best practices recommendation"
  ],
  "performance_analysis": "Analysis of the provided performance data and how it relates to theoretical complexity",
  "algorithm_pattern": "Identify the algorithm pattern used (e.g., 'Two Pointers', 'Dynamic Programming', etc.)"
}}

Think carefully and provide accurate complexity analysis. Be specific about WHY you determined each complexity.
"""
    
    def _call_gemini_api(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Make API call to Gemini using official client with proper error handling"""
        if not self.gemini_model:
            print("❌ Gemini model not initialized")
            return None
            
        try:
            print(f"🚀 Making request to Gemini API")
            print(f"📝 Model: gemini-2.0-flash-exp")
            
            # Create the full prompt with system instructions
            system_instruction = "You are an expert code analyzer specializing in algorithm complexity analysis. Always provide detailed, accurate analysis with clear reasoning. Format responses as valid JSON when requested."
            full_prompt = f"{system_instruction}\n\nUser Request: {prompt}"
            
            # Generate response
            response = self.gemini_model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistent analysis
                    max_output_tokens=3000,  # Enough tokens for detailed analysis
                    top_p=0.9
                )
            )
            
            print(f"✅ API call successful")
            return {
                "choices": [{
                    "message": {
                        "content": response.text
                    }
                }]
            }
            
        except Exception as e:
            print(f"❌ Gemini API request failed: {e}")
            return None
    
    def _parse_analysis_response(self, analysis_text: str) -> Dict[str, Any]:
        """Parse the LLM response to extract structured data"""
        try:
            # Try to find JSON in the response
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = analysis_text[json_start:json_end]
                parsed = json.loads(json_str)
                return parsed
            else:
                # Fallback to regex parsing if no JSON found
                return self._regex_parse_analysis(analysis_text)
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            return self._regex_parse_analysis(analysis_text)
    
    def _regex_parse_analysis(self, text: str) -> Dict[str, Any]:
        """Fallback parsing using regex when JSON parsing fails"""
        import re
        
        # Default values
        result = {
            "complexity": {"time": "O(n)", "space": "O(1)"},
            "suggestions": [],
            "reasoning": text[:500] + "..." if len(text) > 500 else text,
            "code_quality": {"score": 7, "issues": [], "strengths": []},
            "performance_analysis": ""
        }
        
        # Extract complexity information using regex
        complexities = re.findall(r'O\([^)]+\)', text)
        if complexities:
            result["complexity"]["time"] = complexities[0]
            if len(complexities) > 1:
                result["complexity"]["space"] = complexities[1]
        
        # Extract suggestions (look for numbered or bulleted lists)
        suggestions = []
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if (line.startswith('•') or line.startswith('-') or 
                line.startswith('*') or re.match(r'^\d+\.', line)):
                # Clean up the suggestion text
                clean_suggestion = re.sub(r'^[\d\.\-\*•\s]+', '', line).strip()
                if clean_suggestion:
                    suggestions.append(clean_suggestion)
        
        result["suggestions"] = suggestions[:5]  # Top 5 suggestions
        return result
    
    def _fallback_analysis(self, code: str, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide basic heuristic analysis when API fails"""
        
        # Simple heuristic analysis
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        
        # Count loops and nesting
        loop_count = 0
        max_nesting = 0
        current_nesting = 0
        has_recursion = False
        
        for line in lines:
            # Count loop keywords
            if any(keyword in line for keyword in ['for ', 'while ', 'forEach']):
                loop_count += 1
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
            
            # Check for recursion (function calls itself)
            if 'def ' in line:
                func_name = line.split('def ')[1].split('(')[0].strip()
                if any(func_name in l for l in lines):
                    has_recursion = True
            
            # Reset nesting on closing braces/dedentation
            if line in ['}', 'end'] or (line.startswith('return') and current_nesting > 0):
                current_nesting = max(0, current_nesting - 1)
        
        # Determine complexity heuristically
        if max_nesting >= 3:
            time_complexity = "O(n³)"
        elif max_nesting >= 2 or loop_count >= 2:
            time_complexity = "O(n²)"
        elif has_recursion:
            time_complexity = "O(2^n)"
        elif loop_count >= 1:
            time_complexity = "O(n)"
        else:
            time_complexity = "O(1)"
        
        space_complexity = "O(n)" if has_recursion else "O(1)"
        
        return {
            "success": True,
            "analysis": f"Fallback analysis completed. The algorithm appears to have {time_complexity} time complexity with {max_nesting} levels of loop nesting.",
            "complexity": {
                "time": time_complexity,
                "space": space_complexity,
                "time_explanation": f"Based on {max_nesting} levels of nesting and {loop_count} loops",
                "space_explanation": "Based on recursive calls" if has_recursion else "Uses constant extra space"
            },
            "suggestions": [
                "Consider optimizing nested loops if present",
                "Look for opportunities to use more efficient data structures",
                "Check if memoization can improve recursive algorithms",
                "Consider iterative approaches instead of recursion"
            ],
            "reasoning": "Fallback heuristic analysis due to API unavailability",
            "code_quality": {
                "score": 6,
                "issues": ["Unable to perform detailed analysis"],
                "strengths": ["Code structure analyzed"]
            },
            "performance_analysis": "Unable to cross-reference with performance data"
        }


# Global analyzer instance
_gemini_analyzer = None

def get_gemini_analyzer(api_key: str = None) -> SimpleGeminiAnalyzer:
    """Get or create the global Gemini analyzer instance"""
    global _gemini_analyzer
    if _gemini_analyzer is None:
        _gemini_analyzer = SimpleGeminiAnalyzer(api_key)
    return _gemini_analyzer

def analyze_code_with_gemini(
    code: str, 
    language: str, 
    n_vs_time: Optional[Dict[str, Any]] = None,
    n_vs_space: Optional[Dict[str, Any]] = None,
    execution_error: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function for code analysis using Gemini
    
    Args:
        code: Source code to analyze
        language: Programming language
        n_vs_time: Time complexity measurement data
        n_vs_space: Space complexity measurement data  
        execution_error: Any execution errors that occurred
        
    Returns:
        Analysis results dictionary
    """
    analyzer = get_gemini_analyzer(api_key)
    
    # Prepare performance data
    performance_data = {}
    if n_vs_time:
        performance_data["n_vs_time"] = n_vs_time
    if n_vs_space:
        performance_data["n_vs_space"] = n_vs_space
    if execution_error:
        performance_data["execution_error"] = execution_error
    
    return analyzer.analyze_code(code, language, performance_data)


# Test function for development
if __name__ == "__main__":
    # Test the analyzer
    test_code = """
def solution(nums, target):
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
    """
    
    result = analyze_code_with_gemini(
        code=test_code,
        language="python",
        n_vs_time={"10": 0.001, "100": 0.01, "1000": 0.1},
        n_vs_space={"10": 100, "100": 200, "1000": 300}
    )
    
    print("Analysis Result:")
    print(json.dumps(result, indent=2))
