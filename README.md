# tennis-pose

テニスの動作画像から姿勢を分類する機械学習ライブラリ。

骨格キーポイント座標（18関節）を入力とし、ショット種類（フォアハンド・バックハンド・サーブ・レディポジション）を分類する。

---

## 特徴

- 骨格キーポイント座標（36次元）のみを入力とするシンプルなMLP分類器
- PyTorchで実装
- COCO形式のアノテーションJSONに対応

---

## セットアップ

```bash
pip install -r requirements.txt
```

---

## データセット

**Tennis Player Actions Dataset for Human Pose Estimation**

> Chun-Yi Wang, Kalin Guanlun Lai, Hsu-Chun Huang, Wei-Ting Lin
> Mendeley Data, V1, doi: 10.17632/nv3rpsxhhk.1 — CC BY 4.0

各クラス500枚・計2000枚の画像とCOCO形式キーポイントアノテーションを使用。

| クラス | 説明 |
|---|---|
| forehand | フォアハンドストローク |
| backhand | バックハンドストローク |
| serve | サーブ |
| ready_position | レディポジション |

データは `data/` ディレクトリに配置する（Gitには含めない）。

```
data/
├── annotations/
│   ├── forehand.json
│   ├── backhand.json
│   ├── serve.json
│   └── ready_position.json
└── images/
    ├── forehand/
    ├── backhand/
    ├── serve/
    └── ready_position/
```

---

## モデル構成

入力36次元（18関節 × x,y座標）→ MLP → 4クラス出力

```
Linear(36 → 128) → ReLU → Dropout
Linear(128 → 64) → ReLU → Dropout
Linear(64 → 4)
```

---

## 使い方

### 学習

```bash
python -m tennis_pose.train
```

学習済みモデルは `models/` に保存される。

### 推論

```python
from tennis_pose.predict import predict

result = predict("data/annotations/forehand.json", sample_idx=0)
print(result)  # {'class': 'forehand', 'confidence': 0.95}
```

サンプルスクリプト:

```bash
python examples/sample_predict.py
```

---

## 学習結果

| 指標 | 値 |
|---|---|
| テスト精度 | **98.0%** |
| エポック数 | 100 |
| バッチサイズ | 64 |
| オプティマイザ | Adam (lr=0.001) |
| train / val / test 分割 | 70% / 10% / 20% |

学習曲線・混同行列は `models/` に保存される。

![学習曲線](models/learning_curves.png)
![混同行列](models/confusion_matrix.png)

---

## ディレクトリ構成

```
tennis-pose/
├── tennis_pose/
│   ├── dataset.py     # データ読み込み・前処理
│   ├── model.py       # 分類モデル定義
│   ├── train.py       # 学習スクリプト
│   └── predict.py     # 推論スクリプト
├── notebooks/
│   └── explore_dataset.ipynb
├── examples/
│   └── sample_predict.py
├── data/              # データセット（Git管理外）
├── models/            # 学習済みモデル（Git管理外）
├── requirements.txt
└── README.md
```

---

## ライセンス

本プロジェクトのコードは MIT License。
データセットは CC BY 4.0（上記クレジット表記を参照）。
# tennis-pose
# tennis-pose
