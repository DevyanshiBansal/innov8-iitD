import json
import random

class QuestionEngine:
    """
    An adaptive engine to select programming questions based on performance.
    
    This engine loads a graph of problems and their relationships from a JSON file.
    It adaptively selects the next question based on whether the previous question
    was answered correctly, aiming to adjust the difficulty dynamically.
    """

    def __init__(self, filename="data_json.json"):
        """
        Initializes the engine by loading and processing the question data.

        Args:
            filename (str): The path to the JSON file containing the question graph.
        """
        self.nodes_by_id = {}
        self.all_problems = []
        self.relationships = []
        self.difficulty_map = {"Easy": 1, "Medium": 2, "Hard": 3}
        
        self._load_data(filename)
        self._process_data()

    def _load_data(self, filename):
        """Loads the raw data from the specified JSON file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.nodes_by_id = {node['id']: node for node in data.get("nodes", [])}
            self.relationships = data.get("relationships", [])
            print(f"✅ Successfully loaded and parsed '{filename}'.")
        except FileNotFoundError:
            print(f"❌ Error: The file '{filename}' was not found.")
            exit()
        except json.JSONDecodeError:
            print(f"❌ Error: The file '{filename}' is not a valid JSON file.")
            exit()

    def _process_data(self):
        """Processes the loaded data to identify problems and their properties."""
        self.all_problems = [
            node for node in self.nodes_by_id.values() 
            if "Problem" in node.get("labels", [])
        ]
        if not self.all_problems:
            print("❌ Error: No nodes with the label 'Problem' found in the data.")
            exit()

    def get_problem_details(self, problem_id):
        """
        Retrieves the details of a specific problem by its ID.

        Args:
            problem_id (str): The unique ID of the problem.

        Returns:
            dict: The properties of the problem, or None if not found.
        """
        node = self.nodes_by_id.get(problem_id)
        return node.get("properties") if node else None

    def display_problem(self, problem_id):
        """
        Prints a formatted view of the problem statement.

        Args:
            problem_id (str): The ID of the problem to display.
        """
        details = self.get_problem_details(problem_id)
        if not details:
            print("Problem not found.")
            return

        print("\n" + "="*80)
        print(f"Title: {details.get('title', 'N/A')} (Difficulty: {details.get('difficulty', 'N/A')})")
        print(f"Topics: {', '.join(details.get('topics', []))}")
        print("-"*80)
        print("Problem Statement:")
        print(details.get('problem_statement', 'No statement available.'))
        
        # Print examples if they exist
        examples = details.get('examples', [])
        if examples:
            print("\nExamples:")
            for i, ex_str in enumerate(examples, 1):
                try:
                    # The example is stored as a string representation of a dict
                    ex_dict = eval(ex_str)
                    print(f"  Example {i}:")
                    print(f"    Input: {ex_dict.get('input')}")
                    print(f"    Output: {ex_dict.get('output')}")
                    if ex_dict.get('explanation'):
                        print(f"    Explanation: {ex_dict.get('explanation')}")
                except:
                    print(f"  Example {i}: {ex_str}") # Fallback for malformed example strings
        print("="*80 + "\n")


    def choose_next_question(self, last_problem_id, was_correct, asked_ids):
        """
        Adaptively chooses the next question based on performance.

        Args:
            last_problem_id (str): The ID of the problem just attempted.
            was_correct (bool): True if the last problem was solved correctly, False otherwise.
            asked_ids (set): A set of IDs of problems that have already been asked.

        Returns:
            str: The ID of the next problem to ask, or None if no suitable questions are left.
        """
        last_problem_details = self.get_problem_details(last_problem_id)
        if not last_problem_details:
            return self._get_random_unasked_problem(asked_ids)
            
        last_difficulty_score = self.difficulty_map.get(last_problem_details.get("difficulty"), 1)

        # 1. Find all related (similar topic) problems
        related_problem_ids = set()
        for rel in self.relationships:
            if rel.get("type") == "SIMILAR_TOPIC":
                if rel.get("start") == last_problem_id:
                    related_problem_ids.add(rel.get("end"))
                elif rel.get("end") == last_problem_id:
                    related_problem_ids.add(rel.get("start"))
        
        # Filter out already asked questions
        candidate_ids = [pid for pid in related_problem_ids if pid not in asked_ids]
        
        # 2. Filter candidates based on performance
        suitable_candidates = []
        for pid in candidate_ids:
            details = self.get_problem_details(pid)
            if details:
                current_difficulty_score = self.difficulty_map.get(details.get("difficulty"), 1)
                if was_correct and current_difficulty_score >= last_difficulty_score:
                    suitable_candidates.append(pid)
                elif not was_correct and current_difficulty_score <= last_difficulty_score:
                    suitable_candidates.append(pid)
        
        # 3. If a suitable related problem is found, pick one
        if suitable_candidates:
            return random.choice(suitable_candidates)

        # 4. If not, find any unasked problem with an appropriate difficulty
        all_unasked = [p for p in self.all_problems if p['id'] not in asked_ids]
        fallback_candidates = []
        for problem_node in all_unasked:
            details = problem_node.get("properties", {})
            current_difficulty_score = self.difficulty_map.get(details.get("difficulty"), 1)
            
            if was_correct and current_difficulty_score > last_difficulty_score:
                fallback_candidates.append(problem_node['id'])
            elif not was_correct and current_difficulty_score < last_difficulty_score:
                fallback_candidates.append(problem_node['id'])

        if fallback_candidates:
            return random.choice(fallback_candidates)
        
        # 5. If still no options, just pick any random unasked problem
        return self._get_random_unasked_problem(asked_ids)

    def _get_random_unasked_problem(self, asked_ids):
        """Returns a random problem that has not been asked yet."""
        unasked = [p['id'] for p in self.all_problems if p['id'] not in asked_ids]
        return random.choice(unasked) if unasked else None


# --- Main part of the script ---
if __name__ == "__main__":
    # Create an instance of the question engine
    engine = QuestionEngine(filename="data_json.json")
    
    # Simulation setup
    total_questions_to_ask = 5
    asked_question_ids = set()
    
    # Start with a random problem
    current_problem_id = engine._get_random_unasked_problem(asked_question_ids)

    if not current_problem_id:
        print("No problems available to start the interview.")
    else:
        print("🚀 Starting Simulated Technical Interview...")

        for i in range(total_questions_to_ask):
            if not current_problem_id:
                print("\nNo more questions available. Interview over.")
                break

            print(f"\n--- Question {i+1}/{total_questions_to_ask} ---")
            
            # Display the current problem
            engine.display_problem(current_problem_id)
            asked_question_ids.add(current_problem_id)

            # Get simulated performance from the user
            while True:
                performance_input = input("Was the answer correct? (yes/no): ").lower()
                if performance_input in ["yes", "y"]:
                    was_correct = True
                    break
                elif performance_input in ["no", "n"]:
                    was_correct = False
                    break
                else:
                    print("Invalid input. Please enter 'yes' or 'no'.")
            
            # Choose the next question based on performance
            next_problem_id = engine.choose_next_question(
                current_problem_id, was_correct, asked_question_ids
            )
            
            if next_problem_id:
                print(f"Performance recorded. Next question selected.")
            
            current_problem_id = next_problem_id

        if current_problem_id:
             print("\nInterview finished. Thank you!")