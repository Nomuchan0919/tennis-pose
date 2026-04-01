"""
sample_predict.py — 推論・フォームスコアのサンプルスクリプト

使い方:
    python examples/sample_predict.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from tennis_pose.predict import predict
from tennis_pose.form_score import score_from_json

# ---- ショット種類の推論 ----
print("=== ショット種類の推論 ===")
result = predict("data/annotations/forehand.json", sample_idx=0)
print(f"予測クラス: {result['class']}")
print(f"確信度: {result['confidence']:.1%}")

# ---- フォームスコア ----
print()
print("=== フォームスコア ===")

print(score_from_json("data/annotations/forehand.json", shot="forehand", sample_idx=0))
print()
print(score_from_json("data/annotations/backhand.json", shot="backhand", sample_idx=0))
print()
print(score_from_json("data/annotations/serve.json", shot="serve", sample_idx=0))
