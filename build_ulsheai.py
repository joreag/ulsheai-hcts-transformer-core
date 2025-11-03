import os
import subprocess
import sys
import argparse
import importlib.util

# --- Robust Path Setup ---
# Ensures the script can be run from anywhere and still find the correct directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def check_dependencies():
    """Checks if all required packages are installed."""
    print("--- Verifying required packages... ---")
    required_packages = ['torch', 'tqdm', 'nltk']
    missing_packages = []
    
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package)
            
    if not missing_packages:
        print("--- All dependencies are satisfied. ---\n")
        return True
    else:
        print("\n" + "!"*70)
        print("!!! FATAL ERROR: Missing required Python packages.")
        print(f"!!! Missing: {', '.join(missing_packages)}")
        print("!!! Please install them by running the following command:")
        print("!!!")
        print("!!!   pip install -r requirements.txt")
        print("!"*70 + "\n")
        return False


def run_script(command: list, description: str):
    """A helper function to run a python script, stream its output, and handle errors."""
    print("\n" + "#"*70)
    print(f"### STEP: {description}")
    print(f"### Command: {' '.join(command)}")
    print("#"*70)

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=SCRIPT_DIR)
        for line in process.stdout:
            print(line, end='')
        process.wait()
        
        if process.returncode != 0:
            print(f"\n--- FATAL ERROR during: {description} ---")
            return False
            
        print(f"\n--- SUCCESS: {description} complete. ---")
        return True
    except FileNotFoundError:
        print(f"\n--- FATAL ERROR: Command not found. Is Python installed and in your PATH? ---")
        return False
    except Exception as e:
        print(f"\n--- An unexpected error occurred during: {description} ---")
        print(str(e))
        return False

def main(config: dict):
    """
    The main build pipeline for the UlsheAI HCTS-Transformer.
    """
    print("\n" + "="*70)
    print("===   UlsheAI HCTS-Transformer Foundational Mind Compiler   ===")
    print("="*70)

    # <<< NEW: Run the dependency check first >>>
    if not check_dependencies():
        sys.exit(1)

    # --- (The rest of the main function is correct and does not need to be changed) ---
    VOCAB_FILE = "vocab.json"
    PRE_GRAPH_FILE = "pre_graph.json"
    GRAPH_FILE = "knowledge_graph.pkl"
    DATASET_FILE = "grounding_dataset.jsonl"
    FINAL_MODEL_FILE = config['output_model']
    
    # --- Step 1 ... Step 5 (all the run_script calls are correct)
    if not run_script(['python', '-m', 'pipeline.vocabulary_generator', '--output', VOCAB_FILE], "Generate Custom Vocabulary"): sys.exit(1)
    if not run_script(['python', '-m', 'pipeline.ingestion_system', '--curriculum-root', 'curriculum/', '--output', PRE_GRAPH_FILE], "Ingest Raw Curriculum"): sys.exit(1)
    if not run_script(['python', '-m', 'pipeline.knowledge_graph_builder', '--pre-graph-path', PRE_GRAPH_FILE, '--output', GRAPH_FILE], "Build Active Knowledge Graph"): sys.exit(1)
    if not run_script(['python', '-m', 'pipeline.dataset_generator', '--graph-path', GRAPH_FILE, '--output-path', DATASET_FILE], "Generate Grounding Dataset"): sys.exit(1)
    cmd_train = ['python', '-m', 'src.trainer', '--dataset-path', DATASET_FILE, '--vocab-path', VOCAB_FILE, '--output-model', FINAL_MODEL_FILE, '--epochs', str(config['epochs']), '--batch-size', str(config['batch_size']), '--d-model', str(config['d_model'])]
    if not run_script(cmd_train, "Train HCTS-Transformer AI Model"): sys.exit(1)

    print("\n" + "="*70)
    print("===      HCTS-TRANSFORMER FOUNDATIONAL MIND SUCCESSFULLY BUILT      ===")
    print(f"===  Final Model: {FINAL_MODEL_FILE}  ===")
    print("="*70 + "\n")
    
if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="Run the full build pipeline for the HCTS-Transformer.")
        parser.add_argument('--output-model', type=str, default='jarvits_hcts_model.pth', help="Final name for the trained model file.")
        parser.add_argument('--epochs', type=int, default=100, help="Number of epochs to train the model.")
        parser.add_argument('--batch-size', type=int, default=32, help="Batch size for training.")
        parser.add_argument('--d-model', type=int, default=384, help="Dimension of the model (e.g., 256 for small, 384 for medium).")
    
        args = parser.parse_args()
        config = vars(args)

    
        main(config)