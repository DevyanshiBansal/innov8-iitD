"""
Enhanced Adaptive Question Generation System

This system generates problem IDs based on:
- Current problem performance
- Time taken to solve
- Problem type and difficulty
- User's historical performance
- Adaptive difficulty progression

Features:
- Real-time problem generation
- Performance tracking
- Difficulty adaptation
- Knowledge base integration
- Run/Submit functionality
"""

import json
import time
import math
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class ProblemSession:
    """Track user's problem-solving session"""
    problem_id: str
    start_time: float
    end_time: Optional[float] = None
    attempts: int = 0
    runs: int = 0
    is_solved: bool = False
    hints_used: int = 0
    difficulty: str = "Easy"
    topics: List[str] = field(default_factory=list)
    time_taken: Optional[float] = None
    performance_score: float = 0.0

@dataclass
class UserProfile:
    """Track user's overall performance and preferences"""
    username: str
    total_problems_solved: int = 0
    average_time_per_difficulty: Dict[str, float] = field(default_factory=lambda: {"Easy": 300, "Medium": 600, "Hard": 1200})
    strong_topics: List[str] = field(default_factory=list)
    weak_topics: List[str] = field(default_factory=list)
    current_skill_level: float = 1.0  # 1.0 = Beginner, 2.0 = Intermediate, 3.0 = Advanced
    problem_history: List[ProblemSession] = field(default_factory=list)

class AdaptiveQuestionEngine:
    """
    Enhanced Adaptive Question Generation Engine
    
    Generates next problem ID based on:
    - User performance on current problem
    - Time taken vs expected time
    - Problem difficulty progression
    - Topic mastery analysis
    - Adaptive learning algorithms
    """
    
    def __init__(self, knowledge_base_path: str = "knowledge_base"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.problems_data = {}
        self.problem_graph = {}
        self.difficulty_weights = {"Easy": 1.0, "Medium": 2.0, "Hard": 3.0}
        self.topics_graph = {}
        
        # Load knowledge base
        self._load_knowledge_base()
        self._build_problem_relationships()
    
    def _load_knowledge_base(self):
        """Load problems from knowledge base files"""
        try:
            # Load main question data
            ques_data_path = self.knowledge_base_path / "ques_data.json"
            if ques_data_path.exists():
                with open(ques_data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Extract problem info
                    problem_id = f"prob_{hash(data['question']['title']) % 10000}"
                    self.problems_data[problem_id] = {
                        "id": problem_id,
                        "title": data['question']['title'],
                        "problem_statement": data['question']['problemStatement'],
                        "examples": data['question']['examples'],
                        "constraints": data['question'].get('constraints', []),
                        "solution": data.get('solution', {}),
                        "analysis": data.get('analysis', {}),
                        "difficulty": self._estimate_difficulty(data),
                        "topics": self._extract_topics(data),
                        "expected_time": self._estimate_time(data)
                    }
            
            # Load structured problem graph
            graph_data_path = self.knowledge_base_path / "data_json.json"
            if graph_data_path.exists():
                with open(graph_data_path, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                    
                    # Process nodes
                    for node in graph_data.get("nodes", []):
                        if "Problem" in node.get("labels", []):
                            props = node.get("properties", {})
                            self.problems_data[node["id"]] = {
                                "id": node["id"],
                                "title": props.get("title", "Untitled Problem"),
                                "problem_statement": props.get("problem_statement", ""),
                                "examples": props.get("examples", []),
                                "constraints": props.get("constraints", []),
                                "difficulty": props.get("difficulty", "Easy"),
                                "topics": props.get("topics", []),
                                "expected_time": self._estimate_time_from_difficulty(props.get("difficulty", "Easy"))
                            }
                    
                    # Process relationships
                    self.problem_graph = {}
                    for rel in graph_data.get("relationships", []):
                        start_id = rel.get("start")
                        end_id = rel.get("end")
                        rel_type = rel.get("type")
                        
                        if start_id not in self.problem_graph:
                            self.problem_graph[start_id] = []
                        self.problem_graph[start_id].append({
                            "target": end_id,
                            "type": rel_type,
                            "weight": self._get_relationship_weight(rel_type)
                        })
            
            print(f"✅ Loaded {len(self.problems_data)} problems from knowledge base")
            
        except Exception as e:
            print(f"⚠️ Error loading knowledge base: {e}")
            # Create fallback problems
            self._create_fallback_problems()
    
    def _create_fallback_problems(self):
        """Create fallback problems if knowledge base fails to load"""
        fallback_problems = [
            {
                "id": "two_sum",
                "title": "Two Sum",
                "problem_statement": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
                "examples": [{"input": "nums = [2,7,11,15], target = 9", "output": "[0,1]"}],
                "difficulty": "Easy",
                "topics": ["Array", "Hash Table"],
                "expected_time": 900
            },
            {
                "id": "valid_parentheses",
                "title": "Valid Parentheses",
                "problem_statement": "Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.",
                "examples": [{"input": "s = \"()\"", "output": "true"}],
                "difficulty": "Easy",
                "topics": ["String", "Stack"],
                "expected_time": 600
            },
            {
                "id": "longest_substring",
                "title": "Longest Substring Without Repeating Characters",
                "problem_statement": "Given a string s, find the length of the longest substring without repeating characters.",
                "examples": [{"input": "s = \"abcabcbb\"", "output": "3"}],
                "difficulty": "Medium",
                "topics": ["String", "Sliding Window"],
                "expected_time": 1200
            }
        ]
        
        for prob in fallback_problems:
            self.problems_data[prob["id"]] = prob
        
        print(f"✅ Created {len(fallback_problems)} fallback problems")
    
    def _estimate_difficulty(self, data: Dict) -> str:
        """Estimate difficulty based on problem characteristics"""
        # Analyze complexity if available
        analysis = data.get('analysis', {})
        time_complexity = analysis.get('time_complexity', '$O(n)$')
        
        if 'O(1)' in time_complexity or 'O(n)' in time_complexity:
            return "Easy"
        elif 'O(n log n)' in time_complexity or 'O(n^2)' in time_complexity:
            return "Medium"
        else:
            return "Hard"
    
    def _extract_topics(self, data: Dict) -> List[str]:
        """Extract topics from problem data"""
        # Basic topic extraction based on keywords
        problem_text = data['question'].get('problemStatement', '').lower()
        solution_code = data.get('solution', {}).get('code', '').lower()
        
        topics = []
        topic_keywords = {
            "Array": ["array", "list", "index", "element"],
            "String": ["string", "char", "substring", "palindrome"],
            "Hash Table": ["hash", "map", "dict", "key"],
            "Tree": ["tree", "node", "root", "leaf"],
            "Graph": ["graph", "vertex", "edge", "bfs", "dfs"],
            "Dynamic Programming": ["dp", "dynamic", "memo", "cache"],
            "Math": ["math", "number", "digit", "prime"],
            "Stack": ["stack", "push", "pop", "lifo"],
            "Queue": ["queue", "fifo", "deque"],
            "Sorting": ["sort", "merge", "quick", "heap"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in problem_text or keyword in solution_code for keyword in keywords):
                topics.append(topic)
        
        return topics if topics else ["General"]
    
    def _estimate_time(self, data: Dict) -> int:
        """Estimate expected solving time in seconds"""
        difficulty = self._estimate_difficulty(data)
        base_times = {"Easy": 900, "Medium": 1800, "Hard": 3600}  # 15min, 30min, 60min
        return base_times.get(difficulty, 1800)
    
    def _estimate_time_from_difficulty(self, difficulty: str) -> int:
        """Estimate time from difficulty string"""
        base_times = {"Easy": 900, "Medium": 1800, "Hard": 3600}
        return base_times.get(difficulty, 1800)
    
    def _build_problem_relationships(self):
        """Build topic-based relationships between problems"""
        self.topics_graph = {}
        
        for prob_id, prob_data in self.problems_data.items():
            topics = prob_data.get("topics", [])
            for topic in topics:
                if topic not in self.topics_graph:
                    self.topics_graph[topic] = []
                self.topics_graph[topic].append(prob_id)
    
    def _get_relationship_weight(self, rel_type: str) -> float:
        """Get weight for relationship type"""
        weights = {
            "SIMILAR_TOPIC": 0.8,
            "PREREQUISITE": 0.9,
            "FOLLOW_UP": 0.7,
            "SIMILAR_DIFFICULTY": 0.6
        }
        return weights.get(rel_type, 0.5)
    
    def generate_problem_id(
        self, 
        current_problem_id: Optional[str] = None,
        time_taken: Optional[float] = None,
        is_solved: bool = False,
        attempts: int = 1,
        user_profile: Optional[UserProfile] = None,
        session_history: Optional[List[ProblemSession]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate next problem ID based on comprehensive performance analysis
        
        Args:
            current_problem_id: ID of current problem
            time_taken: Time taken to solve (seconds)
            is_solved: Whether current problem was solved
            attempts: Number of attempts made
            user_profile: User's performance profile
            session_history: Recent session history
            
        Returns:
            Tuple of (next_problem_id, problem_data)
        """
        
        if not current_problem_id or current_problem_id not in self.problems_data:
            # First problem - start with easy
            return self._get_starting_problem(user_profile)
        
        current_problem = self.problems_data[current_problem_id]
        performance_metrics = self._analyze_performance(
            current_problem, time_taken, is_solved, attempts, user_profile
        )
        
        # Determine next difficulty level
        next_difficulty = self._determine_next_difficulty(
            current_problem["difficulty"], performance_metrics, user_profile
        )
        
        # Find candidate problems
        candidates = self._find_candidate_problems(
            current_problem, next_difficulty, session_history or []
        )
        
        # Select best problem using adaptive algorithm
        selected_id = self._select_optimal_problem(candidates, performance_metrics, user_profile)
        
        if selected_id and selected_id in self.problems_data:
            return selected_id, self.problems_data[selected_id]
        else:
            # Fallback to random problem of appropriate difficulty
            return self._get_fallback_problem(next_difficulty)
    
    def _get_starting_problem(self, user_profile: Optional[UserProfile]) -> Tuple[str, Dict[str, Any]]:
        """Get appropriate starting problem"""
        if user_profile and user_profile.current_skill_level > 1.5:
            difficulty = "Medium"
        else:
            difficulty = "Easy"
        
        easy_problems = [
            prob_id for prob_id, prob_data in self.problems_data.items()
            if prob_data.get("difficulty") == difficulty
        ]
        
        if easy_problems:
            selected_id = random.choice(easy_problems)
            return selected_id, self.problems_data[selected_id]
        else:
            # Fallback to first available problem
            first_id = list(self.problems_data.keys())[0]
            return first_id, self.problems_data[first_id]
    
    def _analyze_performance(
        self, 
        problem: Dict[str, Any],
        time_taken: Optional[float],
        is_solved: bool,
        attempts: int,
        user_profile: Optional[UserProfile]
    ) -> Dict[str, float]:
        """Analyze performance metrics"""
        metrics = {
            "accuracy_score": 1.0 if is_solved else 0.0,
            "efficiency_score": 1.0,
            "attempt_penalty": 1.0,
            "overall_score": 0.0
        }
        
        # Calculate efficiency score based on time
        if time_taken and problem.get("expected_time"):
            expected_time = problem["expected_time"]
            if time_taken <= expected_time:
                metrics["efficiency_score"] = 1.0
            elif time_taken <= expected_time * 1.5:
                metrics["efficiency_score"] = 0.8
            elif time_taken <= expected_time * 2:
                metrics["efficiency_score"] = 0.6
            else:
                metrics["efficiency_score"] = 0.4
        
        # Calculate attempt penalty
        if attempts > 1:
            metrics["attempt_penalty"] = max(0.3, 1.0 - (attempts - 1) * 0.2)
        
        # Calculate overall score
        metrics["overall_score"] = (
            metrics["accuracy_score"] * 0.5 +
            metrics["efficiency_score"] * 0.3 +
            metrics["attempt_penalty"] * 0.2
        )
        
        return metrics
    
    def _determine_next_difficulty(
        self,
        current_difficulty: str,
        performance_metrics: Dict[str, float],
        user_profile: Optional[UserProfile]
    ) -> str:
        """Determine next problem difficulty"""
        current_level = self.difficulty_weights[current_difficulty]
        performance_score = performance_metrics["overall_score"]
        
        # Adaptive difficulty adjustment
        if performance_score >= 0.8:
            # Excellent performance - increase difficulty
            next_level = min(3.0, current_level + 0.5)
        elif performance_score >= 0.6:
            # Good performance - maintain or slightly increase
            next_level = min(3.0, current_level + 0.2)
        elif performance_score >= 0.4:
            # Fair performance - maintain difficulty
            next_level = current_level
        else:
            # Poor performance - decrease difficulty
            next_level = max(1.0, current_level - 0.3)
        
        # Map back to difficulty string
        if next_level <= 1.3:
            return "Easy"
        elif next_level <= 2.3:
            return "Medium"
        else:
            return "Hard"
    
    def _find_candidate_problems(
        self,
        current_problem: Dict[str, Any],
        target_difficulty: str,
        session_history: List[ProblemSession]
    ) -> List[str]:
        """Find candidate problems for next question"""
        solved_ids = {session.problem_id for session in session_history}
        candidates = []
        
        # First, try to find problems with similar topics
        current_topics = current_problem.get("topics", [])
        
        for topic in current_topics:
            if topic in self.topics_graph:
                for prob_id in self.topics_graph[topic]:
                    if (prob_id not in solved_ids and 
                        prob_id != current_problem["id"] and
                        self.problems_data[prob_id].get("difficulty") == target_difficulty):
                        candidates.append(prob_id)
        
        # If not enough candidates, add problems of target difficulty
        if len(candidates) < 3:
            for prob_id, prob_data in self.problems_data.items():
                if (prob_id not in solved_ids and 
                    prob_id != current_problem["id"] and
                    prob_data.get("difficulty") == target_difficulty):
                    candidates.append(prob_id)
        
        return list(set(candidates))
    
    def _select_optimal_problem(
        self,
        candidates: List[str],
        performance_metrics: Dict[str, float],
        user_profile: Optional[UserProfile]
    ) -> Optional[str]:
        """Select optimal problem from candidates using scoring"""
        if not candidates:
            return None
        
        scored_candidates = []
        
        for candidate_id in candidates:
            score = self._calculate_problem_score(candidate_id, performance_metrics, user_profile)
            scored_candidates.append((candidate_id, score))
        
        # Sort by score and add some randomness
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select from top 3 candidates with weighted probability
        top_candidates = scored_candidates[:3]
        weights = [3, 2, 1][:len(top_candidates)]
        
        return random.choices([c[0] for c in top_candidates], weights=weights)[0]
    
    def _calculate_problem_score(
        self,
        problem_id: str,
        performance_metrics: Dict[str, float],
        user_profile: Optional[UserProfile]
    ) -> float:
        """Calculate suitability score for a problem"""
        problem = self.problems_data[problem_id]
        score = 1.0
        
        # Topic preference scoring
        if user_profile:
            problem_topics = problem.get("topics", [])
            strong_topic_bonus = sum(1 for topic in problem_topics if topic in user_profile.strong_topics) * 0.2
            weak_topic_penalty = sum(1 for topic in problem_topics if topic in user_profile.weak_topics) * 0.1
            score += strong_topic_bonus - weak_topic_penalty
        
        # Add some randomness to prevent monotony
        score += random.uniform(-0.1, 0.1)
        
        return score
    
    def _get_fallback_problem(self, difficulty: str) -> Tuple[str, Dict[str, Any]]:
        """Get fallback problem of specified difficulty"""
        problems_of_difficulty = [
            prob_id for prob_id, prob_data in self.problems_data.items()
            if prob_data.get("difficulty") == difficulty
        ]
        
        if problems_of_difficulty:
            selected_id = random.choice(problems_of_difficulty)
            return selected_id, self.problems_data[selected_id]
        else:
            # Return any available problem
            first_id = list(self.problems_data.keys())[0]
            return first_id, self.problems_data[first_id]
    
    def get_problem_data(self, problem_id: str) -> Optional[Dict[str, Any]]:
        """Get complete problem data by ID"""
        return self.problems_data.get(problem_id)
    
    def update_user_profile(
        self,
        user_profile: UserProfile,
        session: ProblemSession
    ) -> UserProfile:
        """Update user profile based on completed session"""
        user_profile.problem_history.append(session)
        
        if session.is_solved:
            user_profile.total_problems_solved += 1
            
            # Update average time for difficulty
            difficulty = session.difficulty
            if difficulty in user_profile.average_time_per_difficulty:
                current_avg = user_profile.average_time_per_difficulty[difficulty]
                new_avg = (current_avg + (session.time_taken or 0)) / 2
                user_profile.average_time_per_difficulty[difficulty] = new_avg
        
        # Update skill level based on recent performance
        recent_sessions = user_profile.problem_history[-10:]  # Last 10 problems
        if len(recent_sessions) >= 3:
            success_rate = sum(1 for s in recent_sessions if s.is_solved) / len(recent_sessions)
            avg_attempts = sum(s.attempts for s in recent_sessions) / len(recent_sessions)
            
            if success_rate > 0.8 and avg_attempts < 2:
                user_profile.current_skill_level = min(3.0, user_profile.current_skill_level + 0.1)
            elif success_rate < 0.4 or avg_attempts > 3:
                user_profile.current_skill_level = max(1.0, user_profile.current_skill_level - 0.1)
        
        # Update topic strengths/weaknesses
        topic_performance = {}
        for session in user_profile.problem_history:
            for topic in session.topics:
                if topic not in topic_performance:
                    topic_performance[topic] = []
                topic_performance[topic].append(session.is_solved)
        
        user_profile.strong_topics = []
        user_profile.weak_topics = []
        
        for topic, performances in topic_performance.items():
            if len(performances) >= 2:  # Need at least 2 problems to judge
                success_rate = sum(performances) / len(performances)
                if success_rate > 0.7:
                    user_profile.strong_topics.append(topic)
                elif success_rate < 0.3:
                    user_profile.weak_topics.append(topic)
        
        return user_profile
    
    def get_appreciation_message(self, session: ProblemSession, performance_metrics: Dict[str, float]) -> str:
        """Generate appreciation message based on performance"""
        score = performance_metrics["overall_score"]
        
        if score >= 0.9:
            messages = [
                "🎉 Outstanding performance! You solved it efficiently with minimal attempts.",
                "🌟 Excellent work! Your problem-solving skills are impressive.",
                "🚀 Perfect execution! You demonstrated strong algorithmic thinking."
            ]
        elif score >= 0.7:
            messages = [
                "👏 Great job! You solved the problem well.",
                "✨ Good work! Your approach was solid.",
                "🎯 Nice solution! You're making good progress."
            ]
        elif score >= 0.5:
            messages = [
                "👍 Well done! You successfully solved the problem.",
                "💪 Good effort! Keep practicing to improve efficiency.",
                "📈 Progress made! Each problem helps you grow."
            ]
        else:
            messages = [
                "🌱 Keep learning! Every attempt teaches you something new.",
                "💡 Good try! Problem-solving takes practice.",
                "🎓 Learning opportunity! Review the solution and try similar problems."
            ]
        
        return random.choice(messages)


# Test the adaptive engine
if __name__ == "__main__":
    engine = AdaptiveQuestionEngine()
    
    # Create test user profile
    user = UserProfile(username="TestUser")
    
    print("🚀 Testing Adaptive Question Generation System")
    print("=" * 60)
    
    # Start with first problem
    current_id = None
    session_history = []
    
    for i in range(3):
        problem_id, problem_data = engine.generate_problem_id(
            current_problem_id=current_id,
            time_taken=random.randint(300, 1800) if current_id else None,
            is_solved=random.choice([True, False]) if current_id else None,
            attempts=random.randint(1, 3) if current_id else 1,
            user_profile=user,
            session_history=session_history
        )
        
        print(f"\n--- Problem {i+1} ---")
        print(f"ID: {problem_id}")
        print(f"Title: {problem_data.get('title', 'N/A')}")
        print(f"Difficulty: {problem_data.get('difficulty', 'N/A')}")
        print(f"Topics: {', '.join(problem_data.get('topics', []))}")
        print(f"Expected Time: {problem_data.get('expected_time', 0)}s")
        
        # Simulate session
        if current_id:
            session = ProblemSession(
                problem_id=current_id,
                start_time=time.time() - 600,
                end_time=time.time(),
                attempts=random.randint(1, 3),
                runs=random.randint(3, 10),
                is_solved=random.choice([True, False]),
                difficulty=problem_data.get('difficulty', 'Easy'),
                topics=problem_data.get('topics', []),
                time_taken=random.randint(300, 1800)
            )
            session_history.append(session)
            user = engine.update_user_profile(user, session)
        
        current_id = problem_id
    
    print(f"\n📊 User Profile Summary:")
    print(f"Skill Level: {user.current_skill_level:.1f}")
    print(f"Strong Topics: {user.strong_topics}")
    print(f"Problems Solved: {user.total_problems_solved}")