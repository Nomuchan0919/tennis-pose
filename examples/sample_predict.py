"""
sample_predict.py — 推論のサンプルスクリプト

使い方:
    python examples/sample_predict.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from tennis_pose.predict import predict

# フォアハンドのJSONから1番目のサンプルを予測
result = predict("data/annotations/forehand.json", sample_idx=0)
print(f"予測クラス: {result['class']}")
print(f"確信度: {result['confidence']:.1%}")
