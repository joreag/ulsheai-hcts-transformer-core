# UlsheAI Cognitive Tools

This directory contains diagnostic and utility scripts for interacting with a trained `HCTS-Transformer` model.

## `latent_space_mapper.py` - The Cognitive Cartographer

This script is the "fMRI Brain Scanner" for your trained AI. It visualizes the internal conceptual relationships the model has learned by mapping concepts from the Knowledge Graph into the model's "latent space" and calculating their proximity.

### How to Use This Tool

This script is a powerful and flexible command-line tool. After you have successfully run the main `build_ulsheai.py` script and have a trained model file (e.g., `jarvits_hcts_model.pth`), you can use this tool to analyze its "mind."

**To print the full report to the console:**

```bash
python tools/latent_space_mapper.py --model-path jarvits_hcts_model.pth
```

**To save the full report to a JSON file:**

```bash
python tools/latent_space_mapper.py --model-path jarvits_hcts_model.pth --output-file latent_space_report.json
```

**To adjust parameters if you trained a different model size:**

(Note: The model parameters must match the ones used during training.)
```bash
python tools/latent_space_mapper.py --model-path my_small_model.pth --d-model 256 --dim-feedforward 512
```

# Chat with your new AI

## `query.py` ** - The Interactive Interface

This script is a self-contained interface to your newly created Ulshe AI instance.  After you run build_ulsheai.py and create your jarvits_hcts_model.pth, you can immediately start chatting with your new AI with this simple command:
```bash
python tools/query.py --model-path jarvits_hcts_model.pth
```

