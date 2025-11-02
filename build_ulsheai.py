import os
import subprocess
import sys
import argparse

# --- Robust Path Setup ---
# Ensures the script can be run from anywhere and still find the correct directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_script(command: list, description: str):
    """A helper function to run a python script, stream its output, and handle errors."""
    print("\n" + "#"*70)
    print(f"### STEP: {description}")
    print(f"### Command: {' '.join(command)}")
    print("#"*70)

    try:
        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=SCRIPT_DIR # Run all commands from the project root
        )
        for line in process.stdout:
            print(line, end='')
        process.wait()
        
        if process.returncode != 0:
            print(f"\n--- FATAL ERROR during: {description} ---")
            # The full error output will have already been streamed
            return False
            
        print(f"\n--- SUCCESS: {description} complete. ---")
        return True
    except FileNotFoundError:
        print(f"\n--- FATAL ERROR: Command not found. Is Python installed and in your PATH? ---")
        print(f"--- Failed command: {' '.join(command)} ---")
        return False
    except Exception as e:
        print(f"\n--- An unexpected error occurred during: {description} ---")
        print(str(e))
        return False

def main(config: dict):
    """
    The main build pipeline for the UlsheAI HCTS-Transformer.
    Orchestrates all steps from data engineering to AI training.
    """
    print("\n" + "="*70)
    print("===   UlsheAI HCTS-Transformer Foundational Mind Compiler   ===")
    print("="*70 + "\n")

    # Define the file paths for the pipeline artifacts
    VOCAB_FILE = "vocab.json"
    PRE_GRAPH_FILE = "pre_graph.json"
    GRAPH_FILE = "knowledge_graph.pkl"
    DATASET_FILE = "grounding_dataset.jsonl"
    FINAL_MODEL_FILE = config['output_model']

    # --- Step 1: Generate Character-Level Vocabulary ---
    cmd_vocab = ['python', 'pipeline/vocabulary_generator.py', '--output', VOCAB_FILE]
    if not run_script(cmd_vocab, "Generate Custom Vocabulary"):
        sys.exit(1)

    # --- Step 2: Ingest Raw Curriculum into pre_graph.json ---
    # Note: We assume the 'curriculum/' directory exists.
    cmd_ingest = ['python', 'pipeline/ingestion_system.py', '--curriculum-root', 'curriculum/', '--output', PRE_GRAPH_FILE]
    if not run_script(cmd_ingest, "Ingest Raw Curriculum"):
        sys.exit(1)

    # --- Step 3: Build the Active Knowledge Graph (.pkl) from the pre_graph ---
    cmd_kg = ['python', 'pipeline/knowledge_graph_builder.py', '--pre-graph-path', PRE_GRAPH_FILE, '--output', GRAPH_FILE]
    if not run_script(cmd_kg, "Build Active Knowledge Graph"):
        sys.exit(1)

    # --- Step 4: Generate Training Dataset ---
    cmd_dataset = ['python', 'pipeline/dataset_generator.py', '--graph-path', GRAPH_FILE, '--output-path', DATASET_FILE]
    if not run_script(cmd_dataset, "Generate Grounding Dataset"):
        sys.exit(1)

    # --- Step 5: Train the AI Model ---
    cmd_train = [
        'python', 'src/trainer.py',
        '--dataset-path', DATASET_FILE,
        '--vocab-path', VOCAB_FILE,
        '--output-model', FINAL_MODEL_FILE,
        '--epochs', str(config['epochs']),
        '--batch-size', str(config['batch_size']),
        '--d-model', str(config['d_model'])
    ]
    if not run_script(cmd_train, "Train HCTS-Transformer AI Model"):
        sys.exit(1)

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