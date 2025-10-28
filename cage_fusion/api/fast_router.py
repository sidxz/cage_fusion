# cage_fusion/api/app.py
import os
from typing import List, Optional
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from cage_fusion.api.predict_service import CAGEFusionPredictor

CHECKPOINT_BASE_DIR = os.getenv("CHECKPOINT_DIR", "checkpoints/nuisance-pred/")
MODEL_NAME = os.getenv("MODEL_NAME", "Cross-Prompt-Phased-bert-251020")
MODEL_FILE = os.getenv("MODEL_FILE", "latest_checkpoint.pt")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "256"))

CHECKPOINT_DIR = os.path.join(CHECKPOINT_BASE_DIR, MODEL_NAME)
API_CONTROLLER_NAME = os.getenv("API_CONTROLLER_NAME", "cage-fusion-api")

app = FastAPI(title=f"CAGE-Fusion {API_CONTROLLER_NAME}", version="1.0.0")

# CORS if you’ll hit it from browsers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor: Optional[CAGEFusionPredictor] = None


class ItemIn(BaseModel):
    id: Optional[str] = None
    smiles: str


class PredictIn(BaseModel):
    items: List[ItemIn]
    plot_all_attention: bool = False
    attn_plot_dir: Optional[str] = None
    batch_size: Optional[int] = None


@app.on_event("startup")
def startup():
    global predictor
    predictor = CAGEFusionPredictor(
        checkpoint_dir=CHECKPOINT_DIR, model_file_name=MODEL_FILE
    )


@app.get(f"/{API_CONTROLLER_NAME}/health")
def health():
    if predictor is None or not predictor.ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {
        "ready": True,
        "model_name": MODEL_NAME,
        "checkpoint_dir": CHECKPOINT_DIR,
        "model_file": MODEL_FILE,
        "tasks": predictor.tasks,
        "device": str(predictor.device),
    }


@app.post(f"/{API_CONTROLLER_NAME}/predict")
def predict(payload: PredictIn):
    if predictor is None or not predictor.ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    bs = payload.batch_size or BATCH_SIZE
    try:
        # input df get both smiles and ids
        payload_rows = [
            {"SMILES": item.smiles, "Id": item.id} for item in payload.items
        ]
        input_df = pd.DataFrame(payload_rows)
        df = predictor.predict(
            input_df=input_df,
            batch_size=bs,
            plot_all_attention=payload.plot_all_attention,
            attn_plot_dir=payload.attn_plot_dir,
        )
        # return JSON; big results can be CSV-ified on demand
        return {
            "model_name": MODEL_NAME,
            "time_generated": pd.Timestamp.now().isoformat(),
            "n": len(df),
            "columns": list(df.columns),
            "rows": df.astype(object)
            .where(pd.notna(df), None)
            .to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
