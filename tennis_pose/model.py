"""
model.py — 姿勢分類MLPモデル

入力: 骨格キーポイントのx,y座標 (36次元)
出力: 4クラスの分類スコア (forehand / backhand / serve / ready_position)
"""

import torch
import torch.nn as nn


class PoseClassifier(nn.Module):
    """
    テニス姿勢を分類するMLP（多層パーセプトロン）。

    Args:
        input_dim: 入力次元数（デフォルト36 = 18関節 × x,y）
        num_classes: 出力クラス数（デフォルト4）
        dropout: ドロップアウト率（過学習防止）
    """

    def __init__(self, input_dim: int = 36, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
