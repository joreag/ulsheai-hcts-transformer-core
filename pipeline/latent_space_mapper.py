import sys
import os
import json
import torch
import re
from sentence_transformers import util
import pickle

# This block ensures we can import our custom modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from jarvits_modules.hcts_model_architecture import HCTS_Transformer
from jarvits_modules.query_engine import CognitiveNode # We need this for unpickling

class LatentSpaceMapper:
    """
    A diagnostic tool to "poll" the AI's latent space by encoding all
    known concepts and analyzing their similarity.
    """
    def __init__(self, model_path, vocab_path, graph_path):
        print("Initializing Latent Space Mapper...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.vocab = self._load_json(vocab_path)
        self.nodes = self._load_graph(graph_path) # Now loads .pkl
        if not self.nodes:
            raise FileNotFoundError("Knowledge Graph could not be loaded. Halting.")

        self.model = self._load_model(model_path, len(self.vocab))
        self.model.eval()
        self.model.to(self.device)

        self.concept_embeddings = {}
        print(f"Latent Space Mapper is online. Using device: {self.device}")

    def _load_json(self, filepath):
        with open(filepath, 'r') as f: return json.load(f)

    def _load_graph(self, filepath: str):
        try:
            with open(filepath, 'rb') as f: return pickle.load(f)
        except Exception as e: print(f"FATAL ERROR loading .pkl graph: {e}"); return {}

    def _load_model(self, model_path, vocab_size):
        # This instantiation must match the trainer's EXACTLY
        model = HCTS_Transformer(
            vocab_size=vocab_size, d_model=256, nhead_syntax=4,
            nhead_semantic=8, num_syntax_layers=2, num_semantic_layers=2,
            num_reasoning_layers=2, dim_feedforward=512, dropout=0.1,
            # semantic_embedding_dim is not used in this model version
            pad_idx=self.vocab['[PAD]']
        )
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Successfully loaded trained model from '{model_path}'")
        except FileNotFoundError:
            print(f"FATAL ERROR: Model file not found at '{model_path}'."); exit()
        return model

    def _tokenize(self, text, max_length=100):
        """Tokenizes text at the character level."""
        token_ids = [self.vocab.get(char, self.vocab.get('[UNK]')) for char in text.lower()]
        token_ids = [self.vocab['[CLS]']] + token_ids + [self.vocab['[SEP]']]
        padding_len = max_length - len(token_ids)
        token_ids += [self.vocab['[PAD]']] * padding_len
        return torch.tensor(token_ids[:max_length]).unsqueeze(0).to(self.device)

    def map_all_concepts(self):
        """
        [MAP] - Feeds every concept's canonical name through the encoder
        to generate its "thought vector".
        """
        print("\n[MAP] Generating latent space vectors for all known concepts...")
        with torch.no_grad():
            for node_id, node_obj in self.nodes.items():
                canonical_name = node_obj.properties.get('canonical_name')
                if canonical_name:
                    source_tensor = self._tokenize(canonical_name)

                    # --- THE CORRECT WAY TO GET THE "THOUGHT VECTOR" ---
                    # Use the dedicated encode method we designed.
                    # We take the vector for the [CLS] token, which represents the aggregated meaning.
                    memory = self.model.encode(source_tensor)
                    thought_vector = memory[:, 0, :] # Shape: [1, d_model]
                    # --- END CORRECTION ---

                    self.concept_embeddings[canonical_name] = thought_vector

        print(f"  -> Successfully mapped {len(self.concept_embeddings)} concepts.")

    def analyze_concept(self, concept_name: str, top_k=5):
        """
        [ITERATE, CHECK, TRANSFORM] - Analyzes a single concept against all
        others and generates a report.
        """
        if not self.concept_embeddings: print("ERROR: You must run map_all_concepts() first."); return
        print("\n" + "="*60 + f"\n        Latent Space Analysis for: {concept_name}\n" + "="*60)

        target_vector = self.concept_embeddings.get(concept_name)
        if target_vector is None: print(f"  -> Concept '{concept_name}' not found in mapped embeddings."); return

        all_names = list(self.concept_embeddings.keys())
        all_vectors = torch.cat(list(self.concept_embeddings.values()))

        cosine_scores = util.cos_sim(target_vector, all_vectors)[0]
        top_results = torch.topk(cosine_scores, k=min(top_k + 1, len(all_names)))

        print(f"  -> Top {top_k} most closely related concepts in JARVITS's 'mind':")
        for i in range(1, top_k + 1):
            score = top_results.values[i].item()
            name = all_names[top_results.indices[i].item()]
            print(f"    {i}. {name} (Similarity: {score:.4f})")
        print("="*60 + "\n")

if __name__ == '__main__':
    PROJECT_ROOT = '.'
    MODEL_PATH = 'jarvits_qna_ai_v1.0.pth'
    VOCAB_PATH = 'vocab.json'
    GRAPH_PATH = 'knowledge_graph.pkl'

    for path in [MODEL_PATH, VOCAB_PATH, GRAPH_PATH]:
        if not os.path.exists(path): print(f"FATAL ERROR: Necessary file not found: '{path}'"); exit()

    mapper = LatentSpaceMapper(MODEL_PATH, VOCAB_PATH, GRAPH_PATH)
    mapper.map_all_concepts()

    mapper.analyze_concept("The Pythagorean Theorem")
    mapper.analyze_concept("Newton's Law of Motion")
