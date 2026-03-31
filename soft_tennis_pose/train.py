"""
train.py — 学習スクリプト

使い方:
    python -m soft_tennis_pose.train
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from .dataset import TennisPoseDataset, CLASS_NAMES
from .model import PoseClassifier

# ---- 設定 ----
DATA_DIR = "data"
MODEL_SAVE_PATH = "models/pose_classifier.pth"
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
TEST_RATIO = 0.2   # 全体の20%をテストに使用
VAL_RATIO = 0.1    # 全体の10%をバリデーションに使用
RANDOM_SEED = 42


def split_dataset(dataset: TennisPoseDataset):
    """データセットをtrain / val / testに分割する。"""
    indices = list(range(len(dataset)))
    labels = dataset.labels.tolist()

    # train+val と test に分割（クラスバランスを保つ stratify）
    train_val_idx, test_idx = train_test_split(
        indices, test_size=TEST_RATIO, stratify=labels, random_state=RANDOM_SEED
    )
    train_val_labels = [labels[i] for i in train_val_idx]
    val_size = VAL_RATIO / (1 - TEST_RATIO)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size, stratify=train_val_labels, random_state=RANDOM_SEED
    )

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (out.argmax(dim=1) == y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item() * len(y)
            correct += (out.argmax(dim=1) == y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def plot_learning_curves(train_losses, val_losses, train_accs, val_accs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, label="train")
    axes[0].plot(val_losses, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(train_accs, label="train")
    axes[1].plot(val_accs, label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("models/learning_curves.png", dpi=120)
    print("学習曲線を保存: models/learning_curves.png")
    plt.show()


def plot_confusion_matrix(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False)
    plt.title("Confusion Matrix (Test)")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=120)
    print("混同行列を保存: models/confusion_matrix.png")
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # データ読み込みと分割
    print("\n--- データ読み込み ---")
    dataset = TennisPoseDataset(DATA_DIR)
    train_set, val_set, test_set = split_dataset(dataset)
    print(f"train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    # モデル・損失関数・オプティマイザ
    model = PoseClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 学習ループ
    print("\n--- 学習開始 ---")
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_accs.append(tr_acc)
        val_accs.append(vl_acc)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {vl_loss:.4f} acc {vl_acc:.4f}")

    # テスト評価
    print("\n--- テスト評価 ---")
    te_loss, te_acc = evaluate(model, test_loader, criterion, device)
    print(f"test loss: {te_loss:.4f}, test accuracy: {te_acc:.4f}")

    # モデル保存
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nモデルを保存: {MODEL_SAVE_PATH}")

    # 可視化
    print("\n--- 可視化 ---")
    plot_learning_curves(train_losses, val_losses, train_accs, val_accs)
    plot_confusion_matrix(model, test_loader, device)

    return te_acc


if __name__ == "__main__":
    main()
