import json
import torch
import torch.nn as nn
import argparse
import os
import time
import sys
from torch.utils.data import Dataset, DataLoader

# <<< --- ADD THIS BOILERPLATE BLOCK --- >>>
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: For scripts in src/, the project root is the parent directory
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) 
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# <<< --- END OF BOILERPLATE BLOCK --- >>>

# Ensure we can find the architecture file
from src.hcts_transformer_architecture import HCTS_Transformer

class GroundingDataset(Dataset):
    def __init__(self, dataset_filepath, vocab_map, max_length=100):
        self.pairs = self._load_data(dataset_filepath)
        self.vocab = vocab_map
        self.max_length = max_length
    def _load_data(self, filepath):
        try:
            with open(filepath, 'r') as f: return [json.loads(line) for line in f]
        except: return []
    def __len__(self): return len(self.pairs)
    def _tokenize(self, text):
        token_ids = [self.vocab.get(char, self.vocab.get('[UNK]')) for char in str(text).lower()]
        token_ids = [self.vocab['[CLS]']] + token_ids + [self.vocab['[SEP]']]
        padding_len = self.max_length - len(token_ids)
        token_ids += [self.vocab['[PAD]']] * padding_len
        return torch.tensor(token_ids[:self.max_length])
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return self._tokenize(pair['question']), self._tokenize(pair['answer'])

def train_model(config: dict):
    print("--- Starting HCTS-Transformer Training ---")
    
    with open(config['vocab_path'], 'r') as f: vocab = json.load(f)
    VOCAB_SIZE, PAD_IDX = len(vocab), vocab['[PAD]']
    print(f"   Vocabulary size: {VOCAB_SIZE}")
    
    dataset = GroundingDataset(config['dataset_path'], vocab, config['max_seq_length'])
    if not dataset: print("FATAL: Dataset is empty or failed to load."); return
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Training on: {device.upper()}")

    model = HCTS_Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=0.1,
        pad_idx=PAD_IDX
    ).to(device)

    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"   -> Using {torch.cuda.device_count()} GPUs via DataParallel.")
        model = nn.DataParallel(model)

    print(f"   -> Model Initialized. Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    print("\n--- Starting Training Loop ---")
    model.train()
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        total_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()
            prediction = model(src, tgt_input)
            loss = criterion(prediction.view(-1, VOCAB_SIZE), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {total_loss / len(dataloader):.4f}")

    print(f"\n--- Training Complete in { (time.time() - start_time) / 60:.2f} minutes ---")
    
    # Save the underlying model state if using DataParallel
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(model_state, config['output_model'])
    print(f"Educated HCTS model saved to '{config['output_model']}'")

if __name__ == '__main__':
    # A more robust and configurable argument parser
    parser = argparse.ArgumentParser(description="Train a foundational HCTS-Transformer model.")
    
    # Data arguments
    parser.add_argument('--dataset-path', type=str, default='grounding_dataset_qna.jsonl')
    parser.add_argument('--vocab-path', type=str, default='vocab.json')
    parser.add_argument('--output-model', type=str, default='jarvits_hcts_model.pth')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--max-seq-length', type=int, default=100)

    # Model architecture arguments
    parser.add_argument('--d-model', type=int, default=384)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num-encoder-layers', type=int, default=6)
    parser.add_argument('--num-decoder-layers', type=int, default=6)
    parser.add_argument('--dim-feedforward', type=int, default=1024)

    args = parser.parse_args()
    
    # Convert args to a dictionary to pass to the main function
    config = vars(args)
    
    train_model(config)