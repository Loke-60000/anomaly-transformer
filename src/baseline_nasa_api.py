"""Baseline NASA Transformer API.

Loads the *baseline* (non-improved) transformer autoencoder checkpoint and exposes
HTTP endpoints for scoring and threshold-based detection on *windowed* sequences.

Run:
    python baseline_nasa_api.py --checkpoint checkpoints_baseline_nasa/best_model.pt

Then:
    curl http://127.0.0.1:8000/health

Notes:
- Input is expected to be windowed sequences shaped [N, T, F] (NASA: T=50, F=25)
- Scores returned are per-window MSE reconstruction errors.
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Union

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.models.model import create_model


class ScoreRequest(BaseModel):
    # Accept either a single sequence [T, F] or a batch [N, T, F]
    sequence: Optional[List[List[float]]] = None
    sequences: Optional[List[List[List[float]]]] = None


class DetectRequest(BaseModel):
    sequence: Optional[List[List[float]]] = None
    sequences: Optional[List[List[List[float]]]] = None
    threshold_percentile: float = Field(95.0, ge=0.0, le=100.0)


def _to_3d_array(req: Union[ScoreRequest, DetectRequest]) -> np.ndarray:
    if req.sequences is not None:
        arr = np.asarray(req.sequences, dtype=np.float32)
    elif req.sequence is not None:
        arr = np.asarray([req.sequence], dtype=np.float32)
    else:
        raise ValueError("Provide either 'sequence' or 'sequences'")

    if arr.ndim != 3:
        raise ValueError(f"Expected [N, T, F], got shape {arr.shape}")
    return arr


def _load_checkpoint_state_dict(checkpoint_path: str, device: torch.device) -> dict:
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Trainer checkpoints use {model_state_dict: ...}
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    # Fallback (raw state_dict)
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unrecognized checkpoint format")


def build_model(
    checkpoint_path: str,
    model_type: str,
    input_dim: int,
    device: torch.device,
) -> torch.nn.Module:
    # Must match the architecture you trained.
    if model_type == "lightweight":
        model = create_model(
            "lightweight",
            input_dim=input_dim,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.1,
        )
    elif model_type == "full":
        model = create_model(
            "full",
            input_dim=input_dim,
            d_model=128,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
        )
    else:
        raise ValueError("model_type must be 'lightweight' or 'full'")

    state_dict = _load_checkpoint_state_dict(checkpoint_path, device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def score_windows(
    model: torch.nn.Module, windows: np.ndarray, device: torch.device
) -> np.ndarray:
    """Return per-window MSE reconstruction error."""
    x = torch.tensor(windows, dtype=torch.float32, device=device)  # [N,T,F]
    with torch.no_grad():
        recon = model(x)
        # MSE averaged over time+features, per window
        mse = torch.mean((x - recon) ** 2, dim=(1, 2))
    return mse.detach().cpu().numpy()


def build_app(
    checkpoint: str,
    model_type: str = "lightweight",
    input_dim: int = 25,
    device_str: Optional[str] = None,
) -> FastAPI:
    device = (
        torch.device(device_str)
        if device_str
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = build_model(
        checkpoint_path=checkpoint,
        model_type=model_type,
        input_dim=input_dim,
        device=device,
    )

    app = FastAPI(title="Baseline NASA Transformer API")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "checkpoint": checkpoint,
            "model_type": model_type,
            "device": str(device),
        }

    @app.post("/score")
    def score(req: ScoreRequest):
        windows = _to_3d_array(req)
        scores = score_windows(model, windows, device)
        return {
            "n_windows": int(windows.shape[0]),
            "scores": scores.tolist(),
            "mean": float(np.mean(scores)) if scores.size else 0.0,
        }

    @app.post("/detect")
    def detect(req: DetectRequest):
        windows = _to_3d_array(req)
        scores = score_windows(model, windows, device)
        thr = (
            float(np.percentile(scores, req.threshold_percentile))
            if scores.size
            else 0.0
        )
        anomalies = scores > thr
        return {
            "n_windows": int(windows.shape[0]),
            "threshold_percentile": float(req.threshold_percentile),
            "threshold": thr,
            "scores": scores.tolist(),
            "anomalies": anomalies.tolist(),
            "n_anomalies": int(np.sum(anomalies)),
        }

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve baseline NASA transformer API")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints_baseline_nasa/best_model.pt",
        help="Path to baseline checkpoint (best_model.pt)",
    )
    parser.add_argument(
        "--model-type",
        choices=["lightweight", "full"],
        default="lightweight",
        help="Must match how the checkpoint was trained",
    )
    parser.add_argument("--input-dim", type=int, default=25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Late import so importing this module doesn't require uvicorn unless running
    import uvicorn

    global app  # allow `uvicorn baseline_nasa_api:app` too
    app = build_app(
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        input_dim=args.input_dim,
        device_str=args.device,
    )

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
