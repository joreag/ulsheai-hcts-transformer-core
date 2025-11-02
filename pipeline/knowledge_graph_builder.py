import json
import os
import re
import pickle
import math
import sys

# <<< --- ADD THIS BOILERPLATE BLOCK --- >>>
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: For scripts in src/, the project root is the parent directory
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) 
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# <<< --- END OF BOILERPLATE BLOCK --- >>>
# We must import the CognitiveNode class to use it
from src.cognitive_node import CognitiveNode

class KnowledgeGraphBuilder:
    """
    Takes the raw 'pre_graph' data and constructs the final Knowledge Graph.
    This version contains the definitive HYBRID logic for finding solution rules,
    correctly handling all lesson types in our curriculum.
    """
    def __init__(self, pre_graph_path='pre_graph.json'):
        self.raw_lessons = self._load_pre_graph(pre_graph_path)
        self.nodes = {} # This will store our CognitiveNode objects
        print("Knowledge Graph Builder (Definitive Hybrid) initialized.")

    def _load_pre_graph(self, filepath: str):
        try:
            with open(filepath, 'r') as f: return json.load(f)
        except Exception as e:
            print(f"FATAL ERROR: Could not load pre_graph.json from '{filepath}': {e}"); return {}

    def _generate_clean_name(self, label: str) -> str:
        name = label; name = re.sub(r'Level\s-?\d+\s[A-Za-z]+\s-\s', '', name); name = re.sub(r'Verifying\s|Basic\s|Deep\sDive\son\s|Identifying\s', '', name, flags=re.IGNORECASE)
        name = name.replace("Algorithmic Complexity (Big O)", "Big O Notation"); name = name.replace("Logical Fallacies (Affirming the Consequent)", "Affirming the Consequent Fallacy"); return name.strip()

    def _generate_unique_id(self, label: str) -> str:
        s = label.lower(); s = re.sub(r'[^a-z0-9\s-]', '', s); s = re.sub(r'[\s-]+', '_', s).strip('_'); return s or "unnamed_concept"

    def build_graph(self):
        """The main orchestration method to build the graph from raw lessons."""
        if not self.raw_lessons:
            print("No raw lessons found to build graph.")
            return

        print("\n===== Building and Enriching Cognitive Nodes =====")
        # First pass: create all the node objects and find their solutions
        for lesson_key, lesson_data in self.raw_lessons.items():
            node_id = self._generate_unique_id(lesson_data['problem_description'])

            properties = {
                "level": lesson_data.get('level', -1),
                "canonical_name": self._generate_clean_name(lesson_data['problem_description'])
            }
            properties.update(lesson_data.get('properties', {}))

            # --- THE DEFINITIVE HYBRID LEARNING SIMULATION ---
            hypothesis_templates = lesson_data.get("hypothesis_templates", [])
            training_data = lesson_data.get("training_data", [])
            valid_rules = []

            # Heuristic to decide which logic to use
            is_universal_rule_lesson = (len(hypothesis_templates) == 1 and len(training_data) > 1)

            if is_universal_rule_lesson:
                # STRATEGY B: The "Scientific Method" for universal rules like Pythagorean Theorem
                hypothesis = hypothesis_templates[0]
                is_universal = True
                for item in training_data:
                    context = {"math": math, **item}
                    expression_to_test = f"({hypothesis}) == {item.get('expected_result')}"
                    try:
                        if not eval(expression_to_test, context):
                            is_universal = False; break
                    except:
                        is_universal = False; break
                if is_universal:
                    valid_rules.append(hypothesis)
            else:
                # STRATEGY A: The "List of Facts" method for lessons like Newton's Laws
                for i, data_point in enumerate(training_data):
                    if data_point.get("expected_result") is True and i < len(hypothesis_templates):
                        valid_rules.append(hypothesis_templates[i])

            if valid_rules:
                properties["solution_rules"] = valid_rules
            # --- END OF DEFINITIVE LOGIC ---

            self.nodes[node_id] = CognitiveNode(
                node_id=node_id,
                label=lesson_data['problem_description'],
                node_type=lesson_data.get('node_type', 'CognitiveConcept'),
                properties=properties,
                source_lessons=[lesson_key]
            )

        print(f"\n===== Weaving Edges Between {len(self.nodes)} Nodes =====")
        # Second pass: create all the edges
        for lesson_key, lesson_data in self.raw_lessons.items():
            source_node_id = self._generate_unique_id(lesson_data['problem_description'])
            for dep_str in lesson_data.get("dependencies", []):
                target_node_id = self._generate_unique_id(dep_str)
                if source_node_id in self.nodes and target_node_id in self.nodes:
                    self.nodes[source_node_id].add_edge(target_node_id, "depends_on")

    def save_graph(self, output_filename='knowledge_graph.pkl'):
        """Serializes the entire graph of active objects to a file."""
        print(f"\nSerializing Knowledge Graph to '{output_filename}'...")
        with open(output_filename, 'wb') as f:
            pickle.dump(self.nodes, f)
        print("Knowledge Graph build complete.")

if __name__ == '__main__':
    builder = KnowledgeGraphBuilder()
    builder.build_graph()
    builder.save_graph()
    print(f"\n--- Build Summary ---\nTotal Cognitive Nodes Created: {len(builder.nodes)}")
