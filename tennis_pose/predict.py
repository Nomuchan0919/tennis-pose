"""
predict.py — 推論スクリプト

学習済みモデルを使ってショット種類を分類する。
"""

import json
from pathlib import Path

import torch

from .dataset import CLASS_NAMES
from .model import PoseClassifier

MODEL_PATH = "models/pose_classifier.pth"


def _load_model(model_path: str, device: torch.device) -> PoseClassifier:
    model = PoseClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)


def _extract_keypoints(json_path: str, sample_idx: int) -> list[float]:
    with open(json_path, "r") as f:
        data = json.load(f)
    ann = data["annotations"][sample_idx]
    kp = ann["keypoints"]
    # x,y座標のみ（visibilityを除く）
    xy = []
    for i in range(0, len(kp), 3):
        xy.append(float(kp[i]))
        xy.append(float(kp[i + 1]))
    return xy


def predict(json_path: str, sample_idx: int = 0, model_path: str = MODEL_PATH) -> dict:
    """
    1サンプルのショット種類を予測する。

    Args:
        json_path: アノテーションJSONのパス
        sample_idx: 何番目のサンプルを予測するか
        model_path: 学習済みモデルのパス

    Returns:
        {'class': クラス名, 'confidence': 確信度(0〜1)}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(model_path, device)

    xy = _extract_keypoints(json_path, sample_idx)
    x = torch.tensor(xy, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 36)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()

    return {
        "class": CLASS_NAMES[pred_idx],
        "confidence": round(probs[pred_idx].item(), 4),
    }
