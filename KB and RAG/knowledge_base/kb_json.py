import json

def find_problem_nodes_and_relations(filename="data.json"):
    """
    Finds the first two "Problem" nodes and all their associated relationships.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file '{filename}' is not a valid JSON file.")
        return None

    all_nodes = data.get("nodes", [])
    all_relationships = data.get("relationships", [])

    problem_nodes = [
        node for node in all_nodes if "Problem" in node.get("labels", [])
    ]
    
    if len(problem_nodes) < 2:
        print("Could not find at least two 'Problem' nodes in the data.")
        return None
        
    node1 = problem_nodes[0]
    node2 = problem_nodes[1]
    node1_id = node1['id']
    node2_id = node2['id']

    relations_between_them = []
    all_relations_for_node1 = []
    all_relations_for_node2 = []

    for rel in all_relationships:
        start_node = rel.get("start")
        end_node = rel.get("end")

        if start_node == node1_id or end_node == node1_id:
            all_relations_for_node1.append(rel)

        if start_node == node2_id or end_node == node2_id:
            all_relations_for_node2.append(rel)
            
        if {start_node, end_node} == {node1_id, node2_id}:
            relations_between_them.append(rel)

    result = {
        "problem_node_1": node1,
        "problem_node_2": node2,
        "relationships_between_them": relations_between_them,
        "all_relationships_for_node_1": all_relations_for_node1,
        "all_relationships_for_node_2": all_relations_for_node2
    }
    
    return result

def save_data_to_file(data, filename):
    """Saves the given data to a JSON file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Successfully saved the data to '{filename}'")
    except IOError as e:
        print(f"Error writing to file: {e}")

# --- Main part of the script ---
if __name__ == "__main__":
    output_filename = '22.json'
    
    # Step 1: Find the nodes and their relationships
    final_data = find_problem_nodes_and_relations()
    
    # Step 2: If data was found, save it to the output file
    if final_data:
        save_data_to_file(final_data, output_filename)