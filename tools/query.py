import sys
import os
import json
import torch
import argparse

# --- Robust Path Setup ---
# This ensures the script can find the 'src' directory from 'tools/'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Use robust imports assuming the new file structure
from src.hcts_transformer_architecture import HCTS_Transformer

class InteractiveCLI:
    """
    A Command-Line Interface for interacting with a trained HCTS-Transformer model.
    """
    def __init__(self, config: dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- Initializing Jarvits CLI on {self.device.upper()} ---")

        print(f"1. Loading vocabulary from '{config['vocab_path']}'...")
        with open(config['vocab_path'], 'r') as f:
            self.vocab = json.load(f)
            self.rev_vocab = {i: c for c, i in self.vocab.items()}
        self.pad_idx = self.vocab['[PAD]']
        self.sos_idx = self.vocab['[CLS]']
        self.eos_idx = self.vocab['[SEP]']

        print(f"2. Loading HCTS-Transformer model from '{config['model_path']}'...")
        self.model = self._load_model()
        
        print("--- Jarvits Core is online. ---")

    def _load_model(self):
        # Instantiate the model using parameters from config
        model = HCTS_Transformer(
            vocab_size=len(self.vocab),
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            num_encoder_layers=self.config['num_encoder_layers'],
            num_decoder_layers=self.config['num_decoder_layers'],
            dim_feedforward=self.config['dim_feedforward'],
            pad_idx=self.pad_idx
        )
        # Load the saved weights
        checkpoint = torch.load(self.config['model_path'], map_location=self.device)
        model.load_state_dict(checkpoint)
        return model.to(self.device).eval()

    def _tokenize(self, text: str) -> torch.Tensor:
        token_ids = [self.vocab.get(char, self.vocab['[UNK]']) for char in text.lower()]
        token_ids = [self.vocab['[CLS]']] + token_ids + [self.vocab['[SEP]']]
        padding_len = self.config['max_seq_length'] - len(token_ids)
        token_ids += [self.pad_idx] * padding_len
        return torch.tensor(token_ids[:self.config['max_seq_length']]).unsqueeze(0).to(self.device)

    def _decode(self, tensor: torch.Tensor) -> str:
        ids = tensor.squeeze(0).cpu().numpy()
        tokens = [self.rev_vocab.get(_id, '') for _id in ids if _id not in [self.sos_idx, self.eos_idx, self.pad_idx]]
        return ''.join(tokens)

    @torch.no_grad()
    def generate_response(self, prompt: str):
        source_tensor = self._tokenize(prompt)
        target_tensor = torch.tensor([[self.sos_idx]], dtype=torch.long, device=self.device)

        for _ in range(self.config['max_seq_length'] - 1):
            prediction = self.model(source_tensor, target_tensor)
            next_token_id = prediction.argmax(2)[:, -1].item()
            target_tensor = torch.cat([target_tensor, torch.tensor([[next_token_id]], device=self.device)], dim=1)
            if next_token_id == self.eos_idx:
                break
        
        response = self._decode(target_tensor)
        print(f"\nJarvits: {response}\n")

    def run_interactive(self):
        print("\nJarvits Interactive Session. Type 'exit' to end.")
        print("-" * 50)
        while True:
            try:
                user_input = input("> ")
                if user_input.lower() in ['exit', 'quit']:
                    break
                if user_input:
                    self.generate_response(user_input)
            except (KeyboardInterrupt, EOFError):
                break
        print("\nJarvits session ended.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interact with a trained HCTS-Transformer model.")
    
    # File paths
    parser.add_argument('--model-path', type=str, default='jarvits_hcts_model.pth')
    parser.add_argument('--vocab-path', type=str, default='vocab.json')
    
    # Model architecture (must match the trained model)
    parser.add_argument('--d-model', type=int, default=384)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num-encoder-layers', type=int, default=6)
    parser.add_argument('--num-decoder-layers', type=int, default=6)
    parser.add_argument('--dim-feedforward', type=int, default=1024)
    parser.add_argument('--max-seq-length', type=int, default=100)
    
    args = parser.parse_args()
    config = vars(args)

    # Verify necessary files exist
    for path in [config['model_path'], config['vocab_path']]:
        if not os.path.exists(path):
            print(f"FATAL ERROR: Necessary file not found: '{path}'")
            print("Please run the main build.py script first to train a model.")
            sys.exit(1)

    cli = InteractiveCLI(config)
    cli.run_interactive()