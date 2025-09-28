import json
from neo4j import GraphDatabase
from collections import defaultdict
from typing import Dict, List, Any

class Neo4jKBGenerator:
    def __init__(self, uri, username, password):
        """Initialize connection to Neo4j database."""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
    
    def close(self):
        """Close the database connection."""
        self.driver.close()
    
    def clear_database(self):
        """Clear all nodes and relationships from the database."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared successfully!")
    
    def create_constraints_and_indexes(self):
        """Create constraints and indexes for better performance."""
        with self.driver.session() as session:
            # Create unique constraints
            constraints = [
                "CREATE CONSTRAINT problem_title IF NOT EXISTS FOR (p:Problem) REQUIRE p.title IS UNIQUE",
                "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE", 
                "CREATE CONSTRAINT difficulty_level IF NOT EXISTS FOR (d:Difficulty) REQUIRE d.level IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except:
                    pass  # Constraint might already exist
            
            # Create indexes
            indexes = [
                "CREATE INDEX problem_difficulty IF NOT EXISTS FOR (p:Problem) ON (p.difficulty)",
                "CREATE INDEX solution_language IF NOT EXISTS FOR (s:Solution) ON (s.language)",
                "CREATE INDEX complexity_value IF NOT EXISTS FOR (c:TimeComplexity) ON (c.complexity)"
            ]
            
            for index in indexes:
                try:
                    session.run(index)
                except:
                    pass  # Index might already exist
            
            print("Constraints and indexes created!")
    
    def create_problem_node(self, session, problem_data):
        """Create a Problem node."""
        question = problem_data.get("question", {})
        metadata = problem_data.get("metadata", {})
        
        query = """
        CREATE (p:Problem {
            title: $title,
            problem_statement: $statement,
            difficulty: $difficulty,
            topics: $topics,
            constraints: $constraints,
            examples: $examples
        })
        RETURN p
        """
        
        result = session.run(query, 
            title=question.get("title", "Unknown"),
            statement=question.get("problemStatement", ""),
            difficulty=metadata.get("difficulty", "Unknown"),
            topics=metadata.get("topic", []),
            constraints=question.get("constraints", []),
            examples=[str(ex) for ex in question.get("examples", [])]
        )
        
        return result.single()[0]
    
    def create_topic_node(self, session, topic_name):
        """Create a Topic node."""
        query = """
        MERGE (t:Topic {name: $name})
        RETURN t
        """
        result = session.run(query, name=topic_name)
        return result.single()[0]
    
    def create_difficulty_node(self, session, difficulty_level):
        """Create a Difficulty node."""
        query = """
        MERGE (d:Difficulty {level: $level})
        RETURN d
        """
        result = session.run(query, level=difficulty_level)
        return result.single()[0]
    
    def create_solution_node(self, session, solution_data, approach="Standard"):
        """Create a Solution node."""
        query = """
        CREATE (s:Solution {
            language: $language,
            code: $code,
            approach: $approach
        })
        RETURN s
        """
        
        result = session.run(query,
            language=solution_data.get("language", "Unknown"),
            code=solution_data.get("code", ""),
            approach=approach
        )
        
        return result.single()[0]
    
    def create_complexity_nodes(self, session, analysis_data):
        """Create TimeComplexity and SpaceComplexity nodes."""
        tc_node = None
        sc_node = None
        
        if analysis_data:
            # Time Complexity
            tc_query = """
            CREATE (tc:TimeComplexity {
                complexity: $complexity,
                explanation: $explanation
            })
            RETURN tc
            """
            
            tc_result = session.run(tc_query,
                complexity=analysis_data.get("time_complexity", "Unknown"),
                explanation=analysis_data.get("explanation", "")
            )
            tc_node = tc_result.single()[0]
            
            # Space Complexity  
            sc_query = """
            CREATE (sc:SpaceComplexity {
                complexity: $complexity,
                explanation: $explanation
            })
            RETURN sc
            """
            
            sc_result = session.run(sc_query,
                complexity=analysis_data.get("space_complexity", "Unknown"),
                explanation=analysis_data.get("explanation", "")
            )
            sc_node = sc_result.single()[0]
        
        return tc_node, sc_node
    
    def create_relationships(self, session, problem_node, topic_nodes, difficulty_node, 
                           solution_nodes, tc_node, sc_node):
        """Create all relationships for a problem."""
        problem_id = problem_node.element_id
        
        # Problem -> Topics
        for topic_node in topic_nodes:
            session.run("""
                MATCH (p:Problem), (t:Topic)
                WHERE elementId(p) = $p_id AND elementId(t) = $t_id
                CREATE (p)-[:BELONGS_TO_TOPIC]->(t)
            """, p_id=problem_id, t_id=topic_node.element_id)
        
        # Problem -> Difficulty
        if difficulty_node:
            session.run("""
                MATCH (p:Problem), (d:Difficulty)
                WHERE elementId(p) = $p_id AND elementId(d) = $d_id
                CREATE (p)-[:HAS_DIFFICULTY]->(d)
            """, p_id=problem_id, d_id=difficulty_node.element_id)
        
        # Problem -> Solutions
        for solution_node in solution_nodes:
            session.run("""
                MATCH (p:Problem), (s:Solution)
                WHERE elementId(p) = $p_id AND elementId(s) = $s_id
                CREATE (p)-[:HAS_SOLUTION]->(s)
            """, p_id=problem_id, s_id=solution_node.element_id)
        
        # Problem -> Complexities
        if tc_node:
            session.run("""
                MATCH (p:Problem), (tc:TimeComplexity)
                WHERE elementId(p) = $p_id AND elementId(tc) = $tc_id
                CREATE (p)-[:HAS_TIME_COMPLEXITY]->(tc)
            """, p_id=problem_id, tc_id=tc_node.element_id)
        
        if sc_node:
            session.run("""
                MATCH (p:Problem), (sc:SpaceComplexity)
                WHERE elementId(p) = $p_id AND elementId(sc) = $sc_id
                CREATE (p)-[:HAS_SPACE_COMPLEXITY]->(sc)
            """, p_id=problem_id, sc_id=sc_node.element_id)
    
    def create_topic_similarity_relationships(self, session):
        """Create SIMILAR_TOPIC relationships between problems sharing topics."""
        query = """
        MATCH (p1:Problem)-[:BELONGS_TO_TOPIC]->(t:Topic)<-[:BELONGS_TO_TOPIC]-(p2:Problem)
        WHERE p1 <> p2 AND NOT (p1)-[:SIMILAR_TOPIC]-(p2)
        CREATE (p1)-[:SIMILAR_TOPIC {shared_topic: t.name}]->(p2)
        """
        
        result = session.run(query)
        print(f"Created topic similarity relationships!")
    
    def generate_kb_from_json(self, json_file_path):
        """Main method to generate KB from JSON file."""
        # Read JSON data (block-wise for JSON Lines or multi-block format)
        with open(json_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            blocks = [b for b in content.split('\n\n') if b.strip()]
            problems = []
            for i, block in enumerate(blocks):
                try:
                    problems.append(json.loads(block))
                except Exception as e:
                    print(f"Error loading problem block {i+1}: {e}")

        print(f"Loaded {len(problems)} problems from JSON file")
        
        # Clear database and setup
        self.clear_database()
        self.create_constraints_and_indexes()
        
        with self.driver.session() as session:
            # Track created nodes for statistics
            problem_count = 0
            topic_count = 0
            solution_count = 0
            
            # Process each problem
            for i, problem in enumerate(problems):
                if not isinstance(problem, dict):
                    print(f"Skipping problem {i+1}: Not a dict (type={type(problem)})")
                    continue
                try:
                    print(f"Processing problem {i+1}/{len(problems)}: {problem.get('question', {}).get('title', 'Unknown')}")
                    
                    # Create problem node
                    problem_node = self.create_problem_node(session, problem)
                    problem_count += 1
                    
                    # Create topic nodes
                    topic_nodes = []
                    topics = problem.get("metadata", {}).get("topic", [])
                    for topic in topics:
                        topic_node = self.create_topic_node(session, topic)
                        topic_nodes.append(topic_node)
                    
                    # Create difficulty node
                    difficulty = problem.get("metadata", {}).get("difficulty", "Unknown")
                    difficulty_node = self.create_difficulty_node(session, difficulty)
                    
                    # Create solution nodes
                    solution_nodes = []
                    solutions = problem.get("solutions", [])
                    if not solutions and problem.get("solution"):
                        solutions = [problem.get("solution")]
                    
                    for solution in solutions:
                        if isinstance(solution, dict):
                            if "implementations" in solution:
                                # Multiple implementations
                                for impl in solution.get("implementations", []):
                                    sol_node = self.create_solution_node(session, impl, 
                                        solution.get("approach_name", "Standard"))
                                    solution_nodes.append(sol_node)
                                    solution_count += 1
                            else:
                                # Single solution
                                sol_node = self.create_solution_node(session, solution)
                                solution_nodes.append(sol_node)
                                solution_count += 1
                    
                    # Create complexity nodes
                    analysis = problem.get("analysis", {})
                    tc_node, sc_node = self.create_complexity_nodes(session, analysis)
                    
                    # Create all relationships
                    self.create_relationships(session, problem_node, topic_nodes, 
                                           difficulty_node, solution_nodes, tc_node, sc_node)
                    
                except Exception as e:
                    print(f"Error processing problem {i+1}: {str(e)}")
                    continue
            
            # Create topic similarity relationships
            print("Creating topic similarity relationships...")
            self.create_topic_similarity_relationships(session)
            
            # Print statistics
            self.print_statistics(session)
    
    def print_statistics(self, session):
        """Print database statistics."""
        print("\n" + "="*50)
        print("NEO4J KNOWLEDGE BASE STATISTICS")
        print("="*50)
        
        # Count nodes by type
        node_counts = {}
        labels = ["Problem", "Topic", "Difficulty", "Solution", "TimeComplexity", "SpaceComplexity"]
        
        for label in labels:
            result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
            count = result.single()["count"]
            node_counts[label] = count
            print(f"{label} nodes: {count}")
        
        # Count relationships by type
        print(f"\nRelationship counts:")
        rel_types = ["BELONGS_TO_TOPIC", "HAS_DIFFICULTY", "HAS_SOLUTION", 
                    "HAS_TIME_COMPLEXITY", "HAS_SPACE_COMPLEXITY", "SIMILAR_TOPIC"]
        
        for rel_type in rel_types:
            result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
            count = result.single()["count"]
            print(f"{rel_type}: {count}")
        
        # Total counts
        total_nodes = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
        total_rels = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
        
        print(f"\nTotal nodes: {total_nodes}")
        print(f"Total relationships: {total_rels}")
        
        # Show some topics
        print(f"\nSample topics:")
        topics_result = session.run("MATCH (t:Topic) RETURN t.name as name LIMIT 10")
        for record in topics_result:
            print(f"  - {record['name']}")

def main():
    # Neo4j connection settings
    NEO4J_URI = "neo4j://127.0.0.1:7687"  # Default Neo4j bolt URI
    NEO4J_USERNAME = "neo4j"             # Default username
    NEO4J_PASSWORD = "12345678"          # Change this to your Neo4j password
    
    # Input file
    JSON_FILE = "ques_data.json"  # Your JSON file path
    
    try:
        # Create KB generator
        kb_generator = Neo4jKBGenerator(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
        
        # Generate the knowledge base
        kb_generator.generate_kb_from_json(JSON_FILE)
        
        print(f"\nKnowledge base created successfully in Neo4j!")
        print(f"You can now explore it using the Neo4j Browser at http://localhost:7474")
        
        # Close connection
        kb_generator.close()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure Neo4j is running on localhost:7687")
        print("2. Check your username/password credentials")
        print("3. Install neo4j driver: pip install neo4j")
        print("4. Make sure your JSON file exists and is properly formatted")

if __name__ == "__main__":
    main()