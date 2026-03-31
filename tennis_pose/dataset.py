"""
dataset.py — データ読み込みと前処理

COCO形式のJSONアノテーションから骨格キーポイント座標を抽出し、
PyTorchのDatasetクラスとして提供する。
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# クラス名とラベル番号の対応
CLASS_NAMES = ["forehand", "backhand", "serve", "ready_position"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def load_keypoints_from_json(json_path: str, label: int) -> tuple[np.ndarray, np.ndarray]:
    """
    1つのJSONファイルからキーポイント座標とラベルを読み込む。

    Args:
        json_path: アノテーションJSONのパス
        label: クラスラベル（整数）

    Returns:
        features: shape (N, 36) — 各サンプルのキーポイントx,y座標
        labels: shape (N,) — クラスラベル
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    features = []
    for ann in data["annotations"]:
        kp = ann["keypoints"]  # [x0, y0, v0, x1, y1, v1, ..., x17, y17, v17]
        # x,y座標のみ取り出す（visibilityは除く）
        xy = []
        for i in range(0, len(kp), 3):
            xy.append(kp[i])      # x
            xy.append(kp[i + 1])  # y
        features.append(xy)

    features = np.array(features, dtype=np.float32)  # (N, 36)
    labels = np.full(len(features), label, dtype=np.int64)
    return features, labels


class TennisPoseDataset(Dataset):
    """
    テニス姿勢分類データセット。

    4クラス（forehand / backhand / serve / ready_position）の
    骨格キーポイント座標を読み込み、PyTorchのDatasetとして提供する。

    Args:
        data_dir: dataディレクトリのパス（annotations/サブディレクトリを含む）
        normalize: Trueの場合、特徴量を平均0・標準偏差1に正規化する
    """

    def __init__(self, data_dir: str, normalize: bool = True):
        data_dir = Path(data_dir)
        annotations_dir = data_dir / "annotations"

        all_features = []
        all_labels = []

        for class_name, label in CLASS_TO_IDX.items():
            json_path = annotations_dir / f"{class_name}.json"
            features, labels = load_keypoints_from_json(json_path, label)
            all_features.append(features)
            all_labels.append(labels)
            print(f"  {class_name}: {len(features)}サンプル読み込み完了")

        # 全クラスを結合
        self.features = np.concatenate(all_features, axis=0)  # (2000, 36)
        self.labels = np.concatenate(all_labels, axis=0)      # (2000,)

        # 正規化（平均0、標準偏差1）
        if normalize:
            self.mean = self.features.mean(axis=0)
            self.std = self.features.std(axis=0) + 1e-8  # ゼロ除算を防ぐ
            self.features = (self.features - self.mean) / self.std
        else:
            self.mean = None
            self.std = None

        print(f"\n合計: {len(self.features)}サンプル, {self.features.shape[1]}次元")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
