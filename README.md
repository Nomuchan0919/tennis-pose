# tennis-pose

テニスの動作画像から姿勢を分類する機械学習ライブラリ。

骨格キーポイント座標（18関節）を入力とし、ショット種類（フォアハンド・バックハンド・サーブ・レディポジション）を分類する。

---

## 特徴

- 骨格キーポイント座標（36次元）のみを入力とするシンプルなMLP分類器
- PyTorchで実装
- COCO形式のアノテーションJSONに対応
- フォアハンド・バックハンド・サーブの「良いフォーム」をショット別指標でスコアリング（0〜100点）

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

### フォームスコア

インパクト瞬間の静止画から、ショット別の指標を数値化して0〜100点のスコアを返す。

| ショット | 評価指標 |
|---|---|
| フォアハンド | 膝の曲がり / 体重移動（右→左足）/ 腰の回転 |
| バックハンド | 膝の曲がり / 体重移動（左→右足）/ 腰の回転 |
| サーブ | 膝の曲がり / 体の捻り / 肘の高さ / 重心が前 |

```python
from tennis_pose.form_score import score_from_json

# フォアハンド
result = score_from_json("data/annotations/forehand.json", shot="forehand", sample_idx=0)
print(result)
# 【フォアハンド】総合スコア: 55.2 / 100
#   膝の曲がり: 68.3 / 100
#   体重移動: 55.5 / 100
#   腰の回転: 41.6 / 100

# バックハンド
result = score_from_json("data/annotations/backhand.json", shot="backhand", sample_idx=0)
print(result)

# サーブ
result = score_from_json("data/annotations/serve.json", shot="serve", sample_idx=0)
print(result)
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

学習曲線・混同行列は学習実行後に `models/` に保存される。

---

## ディレクトリ構成

```
tennis-pose/
├── tennis_pose/
│   ├── dataset.py     # データ読み込み・前処理
│   ├── model.py       # 分類モデル定義
│   ├── train.py       # 学習スクリプト
│   ├── predict.py     # 推論スクリプト
│   └── form_score.py  # フォームスコアリング
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
