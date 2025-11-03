# UlsheAI HCTS-Transformer Core (v1.0)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Welcome to the open-source release of the foundational **HCTS-Transformer architecture** from UlsheAI. This repository contains the source code for our innovative hierarchical AI model, a complete data engineering pipeline to build a structured knowledge graph, and a sample curriculum to run the entire process from start to finish.

This architecture represents a paradigm shift from monolithic neural networks to a more structured, efficient, and interpretable design inspired by cognitive science.

## What is HCTS? The "Glass Box" Philosophy

The **Hierarchical Cyclical Transformation System (HCTS)** is an architectural philosophy for building AI. Instead of a single, massive, and opaque "black box" network, HCTS organizes an AI into a hierarchy of functionally distinct layers.

In this HCTS-Transformer model, we separate the cognitive tasks of **Syntax**, **Semantics**, and **Reasoning** into their own dedicated processing blocks. This modular "Glass Box" design is the first step towards building AI that is not only powerful but also more understandable and efficient.

This early architecture, while still using standard attention mechanisms, is dramatically smaller and more computationally efficient than comparable monolithic models. It is the foundation upon which our more advanced, proprietary "post-transformer" Chimera architectures are built.

## What is in this Repository?

This repository provides a complete, end-to-end "knowledge pipeline" for building a foundational AI model from scratch.

1.  **Sample Curriculum:** A set of structured `.json` lesson files covering foundational concepts in Logic, Physics, and a basic Dictionary.
2.  **Data Engineering Pipeline:** A series of Python scripts that:
    *   Ingest the raw curriculum (`pipeline/ingestion_system.py`).
    *   Validate the knowledge and build a structured graph of interconnected concepts (`pipeline/knowledge_graph_builder.py`).
    *   Generate a custom, character-level vocabulary (`pipeline/vocabulary_generator.py`).
    *   Create a final Q&A training dataset from the knowledge graph (`pipeline/dataset_generator.py`).
3.  **Core AI Architecture:** The source code for the `HCTS-Transformer` model (`src/hcts_transformer_architecture.py`).
4.  **Training & Build Scripts:** A `trainer.py` script and a master `build_ulsheai.py` orchestrator to run the entire pipeline.

### A Note on Scope: A "DIY Car Kit" for AI, Not a Factory-Built Sedan

It is important to understand what this project is—and what it is not. This is **not** a polished, pre-trained consumer product like ChatGPT.

Think of this repository as a high-performance **"DIY Car Kit" for an AI `model`**.

We are providing you with:

*   The **blueprint** for a revolutionary engine (the HCTS architecture).
*   All the **precision parts** to build it (the source code and pipeline scripts).
*   A **small engine block** to get you started (the sample curriculum).

With this kit, you can assemble your own working AI "model"—and we use the word 'model' in both senses of the word. You can see how all the pieces fit together, and you have a tangible, running engine at the end.

This is an architecture designed to be expanded upon. You can take this kit and turn it into a **dragster**, a **rock crawler**, or a **grand tourer** by designing your own custom parts and providing a bigger engine block (a more comprehensive curriculum).

The purpose of this release is to empower the community of builders, researchers, and hobbyists with a foundational kit for exploring the future of efficient, structured, and interpretable AI.

## Getting Started

### Prerequisites

*   Python 3.8+, We currently build using Python Version 3.12.3
*   We recommend using a virtual environment.
*   We Use Linux for Development, if you use Windows, please make sure your environment is setup correctly for Windows.

### Installation

Clone the repository and install the required dependencies with a single command:

```bash
pip install -r requirements.txt
```

### Quickstart: Build Your First HCTS Model

From the root directory of this project, you can build a new AI model from the included sample curriculum with a single command:

```bash
python build_ulsheai.py
```

This script will automatically perform all the necessary data engineering steps and then train the `HCTS-Transformer` model, saving the final `jarvits_hcts_model.pth` file in the project root.

---

### A Note on Training: The "Blueprint and Key"

This repository includes a basic build and training pipeline to demonstrate the functionality of the architecture. It is the **"blueprint"** for the engine.

In our internal research at UlsheAI, we have found that the true power of hierarchical architectures is unlocked through sophisticated, multi-stage training curriculums—our proprietary **"Artificial Childhood"** pedagogy. These advanced pedagogical techniques are the **"key"** to unlocking the full potential of these models.

We encourage you to experiment! The optimal training methodology will depend on your specific use case. The generic script provided is a powerful starting point. For those interested in leveraging our advanced training, fine-tuning expertise, or our next-generation Chimera architectures for enterprise applications, please visit [ulshe.ai](https://www.ulshe.ai) or contact us.

---

### License

This project is licensed under the **Apache 2.0 License**. See the `LICENSE` file for details.

---

We believe the HCTS paradigm is a crucial step towards a future of more efficient, interpretable, and powerful AI. We are thrilled to share this foundational work with the community and can't wait to see what you build with it.