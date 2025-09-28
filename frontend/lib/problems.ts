export interface Problem {
  id: string
  title: string
  description: string
  constraints?: string
  examples?: Array<{
    input: string
    output: string
    explanation?: string
  }>
  templates: Record<string, string>
}

export const PROBLEMS: Problem[] = [
  {
    id: "two-sum",
    title: "Two Sum",
    description:
      "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
    constraints: "2 ≤ nums.length ≤ 10⁴, -10⁹ ≤ nums[i] ≤ 10⁹, -10⁹ ≤ target ≤ 10⁹",
    examples: [
      {
        input: "nums = [2,7,11,15], target = 9",
        output: "[0,1]",
        explanation: "Because nums[0] + nums[1] == 9, we return [0, 1].",
      },
      {
        input: "nums = [3,2,4], target = 6",
        output: "[1,2]",
      },
    ],
    templates: {
      javascript: `// Two Sum
// Return indices of two numbers adding to target
function twoSum(nums, target) {
  const map = new Map();
  for (let i = 0; i < nums.length; i++) {
    const x = nums[i];
    if (map.has(target - x)) return [map.get(target - x), i];
    map.set(x, i);
  }
  return [];
}
console.log(twoSum([2, 7, 11, 15], 9));
`,
      python: `# Two Sum
# Return indices of two numbers adding to target
def twoSum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
    return []

print(twoSum([2, 7, 11, 15], 9))
`,
      java: `// Two Sum
import java.util.*;

class Main {
    static int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> m = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int x = nums[i];
            if (m.containsKey(target - x)) return new int[]{m.get(target - x), i};
            m.put(x, i);
        }
        return new int[0];
    }
    
    public static void main(String[] args) {
        System.out.println(Arrays.toString(twoSum(new int[]{2, 7, 11, 15}, 9)));
    }
}
`,
      cpp: `// Two Sum
#include <bits/stdc++.h>
using namespace std;

vector<int> twoSum(vector<int> nums, int target) {
    unordered_map<int, int> m;
    for (int i = 0; i < nums.size(); ++i) {
        int x = nums[i];
        if (m.count(target - x)) return {m[target - x], i};
        m[x] = i;
    }
    return {};
}

int main() {
    auto r = twoSum({2, 7, 11, 15}, 9);
    if (!r.empty()) cout << r[0] << "," << r[1] << "\\n";
    return 0;
}
`,
      go: `// Two Sum
package main

import "fmt"

func twoSum(nums []int, target int) []int {
    m := map[int]int{}
    for i, x := range nums {
        if j, ok := m[target-x]; ok {
            return []int{j, i}
        }
        m[x] = i
    }
    return []int{}
}

func main() {
    fmt.Println(twoSum([]int{2, 7, 11, 15}, 9))
}
`,
    },
  },
  {
    id: "reverse-string",
    title: "Reverse String",
    description: "Write a function that reverses a string.",
    examples: [
      {
        input: `"hello"`,
        output: `"olleh"`,
      },
    ],
    templates: {
      javascript: `function reverse(s) {
  return s.split('').reverse().join('');
}
console.log(reverse('hello'));
`,
      python: `def reverse(s):
    return s[::-1]

print(reverse('hello'))
`,
      java: `class Main {
    static String reverse(String s) {
        return new StringBuilder(s).reverse().toString();
    }
    
    public static void main(String[] args) {
        System.out.println(reverse("hello"));
    }
}
`,
      cpp: `#include <bits/stdc++.h>
using namespace std;

string reverseStr(string s) {
    reverse(s.begin(), s.end());
    return s;
}

int main() {
    cout << reverseStr("hello") << "\\n";
    return 0;
}
`,
      go: `package main

import "fmt"

func reverse(s string) string {
    r := []rune(s)
    for i, j := 0, len(r)-1; i < j; i, j = i+1, j-1 {
        r[i], r[j] = r[j], r[i]
    }
    return string(r)
}

func main() {
    fmt.Println(reverse("hello"))
}
`,
    },
  },
  {
    id: "fibonacci",
    title: "Fibonacci",
    description: "Return the nth Fibonacci number, iterative.",
    examples: [
      {
        input: "n = 10",
        output: "55",
      },
    ],
    templates: {
      javascript: `function fib(n) {
  let a = 0, b = 1;
  for (let i = 0; i < n; i++) {
    [a, b] = [b, a + b];
  }
  return a;
}
console.log(fib(10));
`,
      python: `def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

print(fib(10))
`,
      java: `class Main {
    static long fib(int n) {
        long a = 0, b = 1;
        for (int i = 0; i < n; i++) {
            long t = a;
            a = b;
            b = t + b;
        }
        return a;
    }
    
    public static void main(String[] args) {
        System.out.println(fib(10));
    }
}
`,
      cpp: `#include <bits/stdc++.h>
using namespace std;

long long fib(int n) {
    long long a = 0, b = 1;
    for (int i = 0; i < n; i++) {
        long long t = a;
        a = b;
        b = t + b;
    }
    return a;
}

int main() {
    cout << fib(10) << "\\n";
    return 0;
}
`,
      go: `package main

import "fmt"

func fib(n int) int {
    a, b := 0, 1
    for i := 0; i < n; i++ {
        a, b = b, a+b
    }
    return a
}

func main() {
    fmt.Println(fib(10))
}
`,
    },
  },
]

export function getProblem(id: string): Problem | undefined {
  return PROBLEMS.find((p) => p.id === id)
}

export function getProblemTemplate(problemId: string, language: string): string {
  const problem = getProblem(problemId)
  return problem?.templates[language] || getDefaultTemplate(language)
}

export function getDefaultTemplate(lang: string): string {
  const templates: Record<string, string> = {
    javascript: `// JavaScript
function solve() {
  return 42;
}
console.log(solve());
`,
    python: `# Python
def solve():
    return 42
print(solve())
`,
    java: `public class Main {
  public static void main(String[] args) {
    System.out.println(42);
  }
}
`,
    cpp: `#include <bits/stdc++.h>
using namespace std;

int main() {
  cout << 42 << "\\n";
  return 0;
}
`,
    go: `package main

import "fmt"

func main() {
  fmt.Println(42)
}
`,
  }

  return templates[lang] || "// code\n"
}
