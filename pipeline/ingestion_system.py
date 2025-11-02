import os
import json

class IngestionSystem:
    """
    Dynamically discovers and ingests all lesson files from a root
    curriculum directory, creating a unified pre-graph JSON object.
    This system is robust to new categories and requires no code changes
    when new curriculum directories are added.
    """
    def __init__(self, curriculum_root_path: str, output_filepath: str):
        """
        Initializes the ingestion system.

        Args:
            curriculum_root_path (str): The path to the root 'curriculum' directory.
            output_filepath (str): The path to save the final pre_graph.json.
        """
        self.curriculum_root = curriculum_root_path
        self.output_filepath = output_filepath
        self.pre_graph_data = {}
        print("Dynamic Ingestion System initialized.")

    def discover_and_ingest(self):
        """
        Scans all subdirectories of the curriculum root, ingests all .json
        lessons, and builds the pre-graph data structure.
        """
        print(f"1. Scanning for curriculum directories in '{self.curriculum_root}'...")

        if not os.path.isdir(self.curriculum_root):
            print(f"FATAL ERROR: Curriculum root directory not found at '{self.curriculum_root}'")
            return

        # Walk through the entire directory tree starting from the root
        for dirpath, _, filenames in os.walk(self.curriculum_root):
            for filename in filenames:
                if filename.endswith('.json'):
                    # Full path to the lesson file
                    lesson_filepath = os.path.join(dirpath, filename)

                    # Create the relative path key for the pre_graph.json
                    # e.g., 'ontology/L0_the_mobius_principle.json'
                    relative_path_key = os.path.relpath(lesson_filepath, self.curriculum_root)

                    try:
                        with open(lesson_filepath, 'r') as f:
                            lesson_data = json.load(f)
                        self.pre_graph_data[relative_path_key] = lesson_data
                    except json.JSONDecodeError as e:
                        print(f"  -> WARNING: Skipping '{relative_path_key}' due to JSON error: {e}")
                    except Exception as e:
                        print(f"  -> WARNING: Skipping '{relative_path_key}' due to unexpected error: {e}")

        total_lessons = len(self.pre_graph_data)
        if total_lessons > 0:
            print(f"2. Successfully ingested {total_lessons} total lesson files.")
        else:
            print("WARNING: No lesson files were found or ingested.")

        return self.pre_graph_data

    def write_pre_graph_to_file(self):
        """Writes the collected pre-graph data to the specified output file."""
        if not self.pre_graph_data:
            print("No data to write. Halting.")
            return

        print(f"3. Writing unified pre-graph to '{self.output_filepath}'...")
        try:
            with open(self.output_filepath, 'w') as f:
                json.dump(self.pre_graph_data, f, indent=2)
            print("Pre-graph generation complete.")
        except Exception as e:
            print(f"FATAL ERROR: Could not write to '{self.output_filepath}': {e}")


if __name__ == '__main__':
    # This assumes the script is run from the project root directory
    # where the 'curriculum' folder and 'pre_graph.json' are located.
    PROJECT_ROOT = '.'
    CURRICULUM_DIR = os.path.join(PROJECT_ROOT, 'curriculum')
    PRE_GRAPH_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'pre_graph.json')

    print("--- Running Dynamic Curriculum Ingestion ---")
    ingestion_system = IngestionSystem(CURRICULUM_DIR, PRE_GRAPH_OUTPUT_PATH)
    ingestion_system.discover_and_ingest()
    ingestion_system.write_pre_graph_to_file()
