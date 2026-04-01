"""
form_score.py — フォアハンドのフォームスコアリング

静止画（インパクト瞬間）の骨格キーポイントから、
フォームの良し悪しを3指標で数値化して0〜100点のスコアを返す。

【3指標】
1. 膝の曲がり  — 膝関節角度が適切に曲がっているか
2. 体重移動    — 腰の重心が左足寄りに乗っているか
3. 腰の回転    — 腰のラインが横を向いているか（正面向きはNG）

【注意】
静止画のみを使用するため、動作の「順番」（体重移動→腰の回転→インパクト）
は判定できない。インパクト瞬間の姿勢のみを評価する。
"""

import math
import json
from dataclasses import dataclass


# 関節インデックス（COCOフォーマット + neck）
NOSE         = 0
LEFT_EYE     = 1
RIGHT_EYE    = 2
LEFT_EAR     = 3
RIGHT_EAR    = 4
LEFT_SHOULDER  = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW   = 7
RIGHT_ELBOW  = 8
LEFT_WRIST   = 9
RIGHT_WRIST  = 10
LEFT_HIP     = 11
RIGHT_HIP    = 12
LEFT_KNEE    = 13
RIGHT_KNEE   = 14
LEFT_ANKLE   = 15
RIGHT_ANKLE  = 16
NECK         = 17


@dataclass
class FormScoreResult:
    """フォームスコアの結果"""
    total: float          # 総合スコア（0〜100点）
    knee_bend: float      # 膝の曲がりスコア（0〜100点）
    weight_shift: float   # 体重移動スコア（0〜100点）
    hip_rotation: float   # 腰の回転スコア（0〜100点）
    details: dict         # 各指標の計算値（デバッグ用）

    def __str__(self) -> str:
        return (
            f"総合スコア      : {self.total:.1f} / 100\n"
            f"  膝の曲がり    : {self.knee_bend:.1f} / 100\n"
            f"  体重移動      : {self.weight_shift:.1f} / 100\n"
            f"  腰の回転      : {self.hip_rotation:.1f} / 100"
        )


def _get_xy(keypoints: list, idx: int) -> tuple[float, float]:
    """キーポイントリストからx,y座標を取得する。"""
    return keypoints[idx * 3], keypoints[idx * 3 + 1]


def _angle(a: tuple, b: tuple, c: tuple) -> float:
    """
    3点a-b-cのb頂点における角度（度）を返す。
    a, b, c はそれぞれ (x, y) のタプル。
    """
    bax = a[0] - b[0]
    bay = a[1] - b[1]
    bcx = c[0] - b[0]
    bcy = c[1] - b[1]
    dot = bax * bcx + bay * bcy
    norm = math.sqrt(bax**2 + bay**2) * math.sqrt(bcx**2 + bcy**2)
    if norm == 0:
        return 180.0
    cos_angle = max(-1.0, min(1.0, dot / norm))
    return math.degrees(math.acos(cos_angle))


def _score_knee_bend(keypoints: list) -> tuple[float, dict]:
    """
    膝の曲がりスコアを計算する。

    良いフォーム: 膝関節角度が 90°〜150° の範囲（適度に曲がっている）
    悪いフォーム: 170°以上（膝が伸びきっている）

    左右の膝を平均して評価する。
    """
    left_hip    = _get_xy(keypoints, LEFT_HIP)
    left_knee   = _get_xy(keypoints, LEFT_KNEE)
    left_ankle  = _get_xy(keypoints, LEFT_ANKLE)
    right_hip   = _get_xy(keypoints, RIGHT_HIP)
    right_knee  = _get_xy(keypoints, RIGHT_KNEE)
    right_ankle = _get_xy(keypoints, RIGHT_ANKLE)

    left_angle  = _angle(left_hip,  left_knee,  left_ankle)
    right_angle = _angle(right_hip, right_knee, right_ankle)
    avg_angle   = (left_angle + right_angle) / 2

    # 90°以下 → 100点、150°で50点、170°以上 → 0点
    if avg_angle <= 90:
        score = 100.0
    elif avg_angle <= 150:
        score = 100.0 - (avg_angle - 90) / 60 * 50
    elif avg_angle <= 170:
        score = 50.0 - (avg_angle - 150) / 20 * 50
    else:
        score = 0.0

    return score, {"left_knee_angle": left_angle, "right_knee_angle": right_angle}


def _score_weight_shift(keypoints: list) -> tuple[float, dict]:
    """
    体重移動スコアを計算する。

    腰の重心が左右どちらの足寄りかを、x座標の比率で評価する。
    左足寄り（左ankle方向）に乗っているほど高スコア。

    ※ 右利きのフォアハンドを想定。プレイヤーが画像の右方向を向いている前提。
    """
    left_hip    = _get_xy(keypoints, LEFT_HIP)
    right_hip   = _get_xy(keypoints, RIGHT_HIP)
    left_ankle  = _get_xy(keypoints, LEFT_ANKLE)
    right_ankle = _get_xy(keypoints, RIGHT_ANKLE)

    # 腰の重心x座標
    hip_center_x = (left_hip[0] + right_hip[0]) / 2
    left_x  = left_ankle[0]
    right_x = right_ankle[0]

    foot_span = abs(left_x - right_x)
    if foot_span < 1e-6:
        return 50.0, {"hip_center_x": hip_center_x, "ratio": 0.5}

    # 左足を1、右足を0とした比率（左足寄りほど1に近い）
    # left_x < right_x の場合（プレイヤーが右向き）
    if left_x < right_x:
        ratio = (hip_center_x - right_x) / (left_x - right_x)
    else:
        ratio = (hip_center_x - left_x) / (right_x - left_x)

    ratio = max(0.0, min(1.0, ratio))

    # ratio が 0.5〜1.0 → 50〜100点（左足寄り）
    # ratio が 0.0〜0.5 → 0〜50点（右足寄り）
    score = ratio * 100

    return score, {"hip_center_x": hip_center_x, "weight_ratio": ratio}


def _score_hip_rotation(keypoints: list) -> tuple[float, dict]:
    """
    腰の回転スコアを計算する。

    左右のhipを結ぶ線の水平方向からの角度で評価する。
    角度が大きい（腰が横を向いている）ほど高スコア。
    角度が小さい（正面向き）ほど低スコア。

    ※ 2D画像上での近似。カメラ角度により精度は変わる。
    """
    left_hip  = _get_xy(keypoints, LEFT_HIP)
    right_hip = _get_xy(keypoints, RIGHT_HIP)

    dx = abs(left_hip[0] - right_hip[0])
    dy = abs(left_hip[1] - right_hip[1])

    # 腰を結ぶ線の水平方向からの角度
    hip_angle = math.degrees(math.atan2(dy, dx))

    # 0°（完全に水平）→ 正面向き → 低スコア
    # 45°以上 → 横向き → 高スコア
    if hip_angle >= 45:
        score = 100.0
    elif hip_angle >= 15:
        score = (hip_angle - 15) / 30 * 100
    else:
        score = 0.0

    return score, {"hip_line_angle": hip_angle}


def calc_form_score(keypoints: list) -> FormScoreResult:
    """
    フォアハンドのフォームスコアを計算する。

    Args:
        keypoints: COCO形式のキーポイントリスト（54値: 18関節 × [x, y, visibility]）

    Returns:
        FormScoreResult: 各指標のスコアと総合スコア
    """
    knee_score,   knee_details   = _score_knee_bend(keypoints)
    weight_score, weight_details = _score_weight_shift(keypoints)
    hip_score,    hip_details    = _score_hip_rotation(keypoints)

    # 3指標を均等に合算
    total = (knee_score + weight_score + hip_score) / 3

    details = {}
    details.update(knee_details)
    details.update(weight_details)
    details.update(hip_details)

    return FormScoreResult(
        total=round(total, 1),
        knee_bend=round(knee_score, 1),
        weight_shift=round(weight_score, 1),
        hip_rotation=round(hip_score, 1),
        details=details,
    )


def score_from_json(json_path: str, sample_idx: int = 0) -> FormScoreResult:
    """
    JSONファイルから直接フォームスコアを計算する。

    Args:
        json_path: アノテーションJSONのパス
        sample_idx: 何番目のサンプルを評価するか

    Returns:
        FormScoreResult
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    keypoints = data["annotations"][sample_idx]["keypoints"]
    return calc_form_score(keypoints)
