import sys
import os
import json
import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm

# This block ensures we can import from the 'src' directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Use robust imports assuming the new file structure
from src.hcts_transformer_architecture import HCTS_Transformer
from src.cognitive_node import CognitiveNode # Needed for unpickling the graph
from pipeline.knowledge_graph_builder import KnowledgeGraphBuilder # A placeholder for any custom classes in the pkl

class LatentSpaceMapper:
    """
    The UlsheAI 'Cognitive Cartographer'. Visualizes the internal relationships an AI
    has learned by mapping concepts into the model's latent space and calculating
    their proximity.
    """
    def __init__(self, config: dict):
        print("--- Initializing Latent Space Mapper ---")
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   -> Using device: {self.device}")

        self.vocab = self._load_json(config['vocab_path'])
        self.nodes = self._load_graph(config['graph_path'])
        
        self.model = self._load_model()
        self.embeddings = None
        self.concept_names = []

    def _load_json(self, filepath):
        with open(filepath, 'r') as f: return json.load(f)

    def _load_graph(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Knowledge Graph not found at '{filepath}'")
        try:
            with open(filepath, 'rb') as f: return pickle.load(f)
        except Exception as e:
            raise IOError(f"Could not load or unpickle knowledge graph '{filepath}': {e}")
            
    def _load_model(self):
        print(f"   -> Loading model from '{self.config['model_path']}'...")
        # Instantiate the model using parameters from config
        model = HCTS_Transformer(
            vocab_size=len(self.vocab),
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            num_encoder_layers=self.config['num_encoder_layers'],
            num_decoder_layers=self.config['num_decoder_layers'],
            dim_feedforward=self.config['dim_feedforward'],
            pad_idx=self.vocab['[PAD]']
        )
        # Use robust loading logic
        checkpoint = torch.load(self.config['model_path'], map_location=self.device)
        model.load_state_dict(checkpoint) # Assumes final model is just the state_dict
        print("   -> Model loaded successfully.")
        return model.to(self.device).eval()

    def _tokenize(self, text: str) -> torch.Tensor:
        token_ids = [self.vocab.get(char, self.vocab['[UNK]']) for char in text.lower()]
        token_ids = [self.vocab['[CLS]']] + token_ids + [self.vocab['[SEP]']]
        padding_len = self.config['max_seq_length'] - len(token_ids)
        token_ids += [self.vocab['[PAD]']] * padding_len
        return torch.tensor(token_ids[:self.config['max_seq_length']]).unsqueeze(0).to(self.device)

    def map_all_concepts(self):
        print("\n[MAP] Generating latent space vectors for all concepts in the Knowledge Graph...")
        # Get all concept names from the nodes of the graph
        all_concepts = [node.properties.get('canonical_name', node.label) for node in self.nodes.values()]
        embeddings_list = []
        
        with torch.no_grad():
            for name in tqdm(all_concepts, desc="Mapping Concepts"):
                question = f"what is {name.lower()}?"
                source_tensor = self._tokenize(question)
                try:
                    # model.encode() returns the memory tensor from the encoder
                    embedding = self.model.encode(source_tensor).mean(dim=1)
                    embeddings_list.append(embedding)
                    self.concept_names.append(name)
                except Exception as e:
                    tqdm.write(f"  -> WARNING: Could not generate embedding for '{name}': {e}")

        self.embeddings = torch.cat(embeddings_list, dim=0)
        print(f"--- Successfully mapped {len(self.concept_names)} concepts. ---")

    def analyze_and_output(self):
        if self.embeddings is None:
            print("No concepts mapped. Please run map_all_concepts() first.")
            return

        print(f"\n[ANALYZE] Calculating conceptual similarities...")
        # Efficiently calculate the entire similarity matrix in one go
        similarity_matrix = F.cosine_similarity(self.embeddings.unsqueeze(1), self.embeddings.unsqueeze(0), dim=2)
        top_k = self.config['top_k']
        top_results = torch.topk(similarity_matrix, k=top_k + 1, dim=1)

        full_report = {}
        for i, name in enumerate(self.concept_names):
            related_concepts = []
            for j in range(top_k + 1): # Loop to k+1 to find itself
                score = top_results.values[i, j].item()
                idx = top_results.indices[i, j].item()
                related_name = self.concept_names[idx]
                if related_name != name:
                    related_concepts.append({"concept": related_name, "similarity": round(score, 4)})
            full_report[name] = related_concepts
        
        if self.config['output_file']:
            print(f"--- Writing full analysis to '{self.config['output_file']}'... ---")
            with open(self.config['output_file'], 'w') as f:
                json.dump(full_report, f, indent=2)
            print("--- Analysis complete. ---")
        else:
            print("\n" + "#"*70)
            print("###         COMPREHENSIVE LATENT SPACE ANALYSIS         ###")
            print("#"*70)
            for name, relations in sorted(full_report.items()):
                print(f"\n>> Concept: {name}")
                print(f"   Top {top_k} most closely related concepts:")
                for i, rel in enumerate(relations):
                    print(f"    {i+1}. {rel['concept']} (Similarity: {rel['similarity']:.4f})")
            print("\n--- Analysis complete. ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the UlsheAI Cognitive Cartographer (Latent Space Mapper).")
    
    # File paths
    parser.add_argument('--model-path', type=str, default='jarvits_hcts_model.pth')
    parser.add_argument('--vocab-path', type=str, default='vocab.json')
    parser.add_argument('--graph-path', type=str, default='knowledge_graph.pkl')
    parser.add_argument('--output-file', type=str, default=None, help="Optional: Path to save the full report as a JSON file.")
    
    # Model architecture (must match the trained model)
    parser.add_argument('--d-model', type=int, default=384)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num-encoder-layers', type=int, default=6)
    parser.add_argument('--num-decoder-layers', type=int, default=6)
    parser.add_argument('--dim-feedforward', type=int, default=1024)
    parser.add_argument('--max-seq-length', type=int, default=100)

    # Analysis parameters
    parser.add_argument('--top-k', type=int, default=5)
    
    args = parser.parse_args()
    config = vars(args)

    try:
        mapper = LatentSpaceMapper(config)
        mapper.map_all_concepts()
        mapper.analyze_and_output()
    except (FileNotFoundError, IOError) as e:
        print(f"\nFATAL ERROR: {e}")
        sys.exit(1)