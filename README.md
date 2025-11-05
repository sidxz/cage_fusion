# CAGE-Fusion: Deep Learning for Nuisance Compound Detection

CAGE-Fusion is a **deep learning framework** for identifying nuisance compounds—such as aggregators, luciferase inhibitors, reactive, and promiscuous molecules—that often cause false positives in early drug discovery.  
It integrates **graph neural networks (GNNs)**, **SMILES-based transformers**, and **physicochemical descriptors** through a **gated co-attention mechanism**, enabling context-aware and interpretable predictions.

> ⚠️ This repository hosts the core code and API for CAGE-Fusion.  
> Detailed architectural and dataset information will be released upon publication of the corresponding manuscript.

---

## Overview

CAGE-Fusion leverages **multimodal molecular representations** to capture both topological and sequence-level chemical context:
- **Graph Encoder:** Directed Message Passing Neural Network (D-MPNN)
- **SMILES Encoder:** Transformer-based chemical language model (ChemBERTa backbone)
- **Co-Attention Fusion:** Enables iterative alignment between atoms and SMILES tokens
- **Descriptor Stream:** Adds global RDKit-derived 2D physicochemical features

The model outputs **probabilities for multiple nuisance classes** and provides **interpretable attention maps** that highlight key substructures influencing the prediction.

---

## API Deployment (Docker)

To serve the CAGE-Fusion model as an inference API, use the provided `docker-compose.yml`.

### Example `docker-compose.yml`
```yaml
services:
  cage_fusion_nuisance_api:
    build: .
    volumes:
      - ./checkpoints:/checkpoints
    ports:
      - "10002:10002"
    environment:
      CHECKPOINT_DIR: /checkpoints/nuisance-pred
      MODEL_NAME: Cross-Prompt-Phased-bert-251020
      MODEL_FILE: latest_checkpoint.pt
      BATCH_SIZE: "24"
    networks:
      - daikon-be-net

networks:
  daikon-be-net:
    external: true
    name: daikon-be-net
```

### Running the API
```bash
docker compose up --build -d
```

Once started, the FastAPI service runs at:

```
http://localhost:10002/docs
```

---

## Entrypoint Script

The container uses a startup script that:
- Verifies GPU availability
- Caches Hugging Face transformer models
- Validates local checkpoints
- Switches transformers to offline mode
- Launches the FastAPI service with Uvicorn


---

## Environment Variables

| Variable | Description | Default |
|-----------|-------------|----------|
| `CHECKPOINT_DIR` | Path to model checkpoint directory | `/checkpoints/nuisance-pred` |
| `MODEL_NAME` | Name of the checkpoint subdirectory | `Cross-Prompt-Phased-bert-251020` |
| `MODEL_FILE` | Model checkpoint filename | `latest_checkpoint.pt` |
| `PORT` | API port | `10002` |
| `WORKERS` | Number of Uvicorn workers | `1` |


---

## Requirements

- Docker with GPU support (optional)
- Mounted model checkpoint under `/checkpoints/nuisance-pred`

---


## License

This project is distributed for **academic and non-commercial use** only.  
See the `LICENSE` file for terms.

---


