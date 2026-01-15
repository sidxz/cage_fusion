# CAGE-Fusion  
**Deep Learning for Assay Nuisance Compound Detection Using a Gated Co-Attention Graph Embedding Model (CAGE-Fusion)**

CAGE-Fusion is a **general-purpose deep learning framework** for molecular property prediction that integrates **graph-based**, **sequence-based**, and **descriptor-based** molecular representations through a **gated co-attention mechanism**.

The architecture is designed to enable **bidirectional information exchange** between molecular graphs and SMILES sequences, producing chemically coherent representations that are both **highly predictive** and **interpretable**. While CAGE-Fusion is applicable to a wide range of molecular clasification tasks, it is demonstrated here through a comprehensive **assay nuisance compound detection** case study.


> 
> **ChemRxiv:** https://chemrxiv.org/engage/chemrxiv/article-details/69612bd3fc9dac0f37ebd868
> 
> **Datasets:** https://zenodo.org/records/17118024
> 
> **Model Weights:** https://files.orca-03.biobio.tamu.edu/model-weights/cage-fusion/nuisance/
> 
> **Inference Web Server:** https://studio.orca-03.biobio.tamu.edu/

---

## Key Contributions

- **Multimodal molecular learning framework** combining:
  - Molecular graphs (D-MPNN)
  - SMILES sequences (Transformer / ChemBERTa)
  - Optional physicochemical descriptors (RDKit)
- **Gated co-attention mechanism** enabling iterative cross-modal refinement
- **Interpretable attention maps** highlighting chemically relevant substructures
- **Production-ready deployment** via Docker and FastAPI


---

## Assay Nuisance Compound Detection

CAGE-Fusion is applied to the detection of **assay nuisance compounds**—molecules that generate misleading signals in biochemical or cell-based assays.

### Supported Nuisance Classes
- **Aggregators**
- **Luciferase inhibitors**
- **Reactive compounds**
- **Promiscuous / frequent hitters**
- 
---

## Inference API Deployment (Docker)

CAGE-Fusion can be deployed as a **FastAPI-based inference service** using Docker.

### Example `docker-compose.yml`

To serve the CAGE-Fusion model as an inference API, use the provided `docker-compose.yml`.

### Example `docker-compose.yml`
```yaml
services:
  cage_fusion_nuisance_api:
    build: .
    volumes:
      - ./checkpoints:/checkpoints
      - ./logs:/logs
      - ./pred_results:/pred_results
    ports:
      - "10002:10002"
    environment:
      CHECKPOINT_DIR: /checkpoints/nuisance-pred
      CAGE_FUSION_LOG_DIR: /logs
      PRED_RES_ROOT_DIR: /pred_results
      MODEL_NAME: DeepChem-ChemBERTa-77M-MTR-CoAttn-1-cross-aux-fgprompt
      MODEL_FILE: latest_checkpoint.pt
      BATCH_SIZE: "24"

```

### Running the API
```bash
docker compose up --build -d
```

FastAPI service runs at:

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
| `MODEL_NAME` | Name of the checkpoint subdirectory | `DeepChem-ChemBERTa-77M-MTR-CoAttn-1-cross-aux-fgprompt` |
| `MODEL_FILE` | Model checkpoint filename | `latest_checkpoint.pt` |
| `PORT` | API port | `10002` |
| `WORKERS` | Number of Uvicorn workers | `1` |


---

## Requirements

- Docker with GPU support (optional)
- Mounted model checkpoint under `/checkpoints/nuisance-pred`


## License

This project is distributed for **academic and non-commercial use** only.  
See the `LICENSE` file for terms.

---


