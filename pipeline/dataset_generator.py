import json
import random
import os
import pickle
import argparse
import sys # <<< MODIFICATION: Import sys

# <<< MODIFICATION START: The definitive fix for the ModuleNotFoundError
# This block ensures the script can always find its sibling modules like cognitive_node
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root that contains the 'jarvits_modules' package
PROJECT_ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.append(PROJECT_ROOT_DIR)

# Now, we can use a robust, absolute import
from src.cognitive_node import CognitiveNode
# <<< MODIFICATION END

class GroundingDatasetGenerator:
    def __init__(self, graph_filepath: str):
        self.nodes = self._load_graph(graph_filepath)
        print("Grounding Dataset Generator (Smart Q&A) initialized.")

    def _load_graph(self, filepath: str):
        if not os.path.exists(filepath):
            print(f"FATAL ERROR: Knowledge Graph not found at '{filepath}'")
            return {}
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"FATAL ERROR: Could not load knowledge graph '{filepath}': {e}"); return {}

    def generate_question_variations(self, concept_name: str, question_type: str = "general") -> list:
        name = concept_name.lower()
        if question_type == "definition":
            return [f"what is the definition of {name}?", f"define {name}"]
        elif question_type == "summary":
            return [f"what is a summary of {name}?", f"summarize {name} for me"]
        else: # general / rules
            return [f"what is {name}?", f"tell me about {name}", f"explain the principles of {name}"]

    def generate_qa_pairs(self) -> list:
        """
        The definitive generator. Creates multiple, comprehensive Q&A pairs
        by intelligently reading from the properties of each node.
        """
        if not self.nodes:
            print("Knowledge graph is empty. No Q&A pairs to generate.")
            return []

        qa_pairs = []
        # Ensure we are iterating over a copy of the items, which is safer
        for node_id, node_obj in list(self.nodes.items()):
            properties = node_obj.properties
            
            # Use a robust way to get the primary name for forming questions
            # The label on the node object itself is the most reliable source.
            display_name = node_obj.label
            if not display_name:
                continue

            # --- This logic is now cleaner and more robust ---
            if 'definition' in properties:
                for q in self.generate_question_variations(display_name, "definition"):
                    qa_pairs.append({"question": q, "answer": f"the definition of {display_name.lower()} is: {str(properties['definition']).lower()}."})
            
            if 'summary' in properties:
                 for q in self.generate_question_variations(display_name, "summary"):
                    qa_pairs.append({"question": q, "answer": f"a summary of {display_name.lower()} is: {str(properties['summary']).lower()}."})

            if 'solution_rules' in properties and properties['solution_rules']:
                 for q in self.generate_question_variations(display_name, "general"):
                    qa_pairs.append({"question": q, "answer": f"the principles of {display_name.lower()} are: {', '.join(map(str, properties['solution_rules']))}."})
            
            # Generate from all other available properties for primitives and other concepts
            for key, value in properties.items():
                # Avoid re-generating from keys we've already handled
                if key not in ['definition', 'summary', 'solution_rules', 'canonical_name', 'word', 'components']:
                    question = f"what is the {key} of {display_name.lower()}?"
                    answer = str(value).lower()
                    qa_pairs.append({"question": question, "answer": answer})

            # Logic for Spelling Bee curriculum
            if 'word' in properties and 'components' in properties:
                word_to_spell = properties['word']
                components = properties['components']
                
                spelling_question = f"how do you spell {word_to_spell}?"
                spelling_answer = " ".join(components)
                qa_pairs.append({"question": spelling_question, "answer": spelling_answer})

                components_str = " ".join(components)
                formation_question = f"what word do the letters {components_str} make?"
                formation_answer = word_to_spell
                qa_pairs.append({"question": formation_question, "answer": formation_answer})

        return qa_pairs

    def write_dataset_to_file(self, qa_pairs: list, output_filepath: str):
        print(f"Writing {len(qa_pairs)} Q&A pairs to '{output_filepath}'...")
        random.shuffle(qa_pairs)
        output_dir = os.path.dirname(output_filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        # --- END OF FIX ---
        with open(output_filepath, 'w') as f:
            for pair in qa_pairs:
                f.write(json.dumps(pair) + '\n')
        print("Dataset generation complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a Q&A grounding dataset from a Knowledge Graph.")
    parser.add_argument('--graph-path', default='knowledge_graph.pkl', help="Path to the input knowledge_graph.pkl file.")
    parser.add_argument('--output-path', default='grounding_dataset_qna.jsonl', help="Path for the output .jsonl dataset file.")
    args = parser.parse_args()

    print("--- Running Smart Q&A Dataset Generation ---")
    generator = GroundingDatasetGenerator(graph_filepath=args.graph_path)
    qa_pairs = generator.generate_qa_pairs()

    if qa_pairs:
        generator.write_dataset_to_file(qa_pairs, output_filepath=args.output_path)
        print(f"\n--- Dataset Generation Success ---")
        print(f"Total Q&A pairs generated: {len(qa_pairs)}")
        print("\nFirst 5 Q&A pairs (shuffled):")
        for pair in qa_pairs[:5]:
            print(json.dumps(pair, indent=2))
