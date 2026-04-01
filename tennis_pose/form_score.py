"""
form_score.py — テニスショット別フォームスコアリング

静止画（インパクト瞬間）の骨格キーポイントから、
フォームの良し悪しを指標ごとに数値化して0〜100点のスコアを返す。

【注意】
静止画のみを使用するため、動作の「順番」（体重移動→腰の回転→インパクト）
は判定できない。インパクト瞬間の姿勢のみを評価する。

【ショット別の評価指標】
フォアハンド : 膝の曲がり / 体重移動（右→左足）/ 腰の回転
バックハンド : 膝の曲がり / 体重移動（左→右足）/ 腰の回転
サーブ       : 膝の曲がり / 体の捻り（肩と腰のねじれ）/ 肘の高さ / 重心が前
"""

import math
import json
from dataclasses import dataclass, field


# 関節インデックス（COCOフォーマット + neck）
NOSE           = 0
LEFT_EYE       = 1
RIGHT_EYE      = 2
LEFT_EAR       = 3
RIGHT_EAR      = 4
LEFT_SHOULDER  = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW     = 7
RIGHT_ELBOW    = 8
LEFT_WRIST     = 9
RIGHT_WRIST    = 10
LEFT_HIP       = 11
RIGHT_HIP      = 12
LEFT_KNEE      = 13
RIGHT_KNEE     = 14
LEFT_ANKLE     = 15
RIGHT_ANKLE    = 16
NECK           = 17


@dataclass
class FormScoreResult:
    """フォームスコアの結果"""
    shot: str             # ショット種類
    total: float          # 総合スコア（0〜100点）
    scores: dict          # 各指標のスコア {指標名: スコア}
    details: dict = field(default_factory=dict)  # 計算値（デバッグ用）

    def __str__(self) -> str:
        lines = [f"【{self.shot}】総合スコア: {self.total:.1f} / 100"]
        for name, score in self.scores.items():
            lines.append(f"  {name}: {score:.1f} / 100")
        return "\n".join(lines)


def _get_xy(keypoints: list, idx: int) -> tuple:
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


# ---- 共通指標 ----

def _score_knee_bend(keypoints: list) -> tuple:
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

    if avg_angle <= 90:
        score = 100.0
    elif avg_angle <= 150:
        score = 100.0 - (avg_angle - 90) / 60 * 50
    elif avg_angle <= 170:
        score = 50.0 - (avg_angle - 150) / 20 * 50
    else:
        score = 0.0

    return score, {"left_knee_angle": left_angle, "right_knee_angle": right_angle}


def _score_hip_rotation(keypoints: list) -> tuple:
    """
    腰の回転スコアを計算する。

    左右のhipを結ぶ線が水平から傾いているほど横向き（良いフォーム）。
    0°（水平）→ 正面向き → 低スコア
    45°以上 → 横向き → 高スコア
    ※ 2D画像上での近似。カメラ角度により精度が変わる。
    """
    left_hip  = _get_xy(keypoints, LEFT_HIP)
    right_hip = _get_xy(keypoints, RIGHT_HIP)

    dx = abs(left_hip[0] - right_hip[0])
    dy = abs(left_hip[1] - right_hip[1])
    hip_angle = math.degrees(math.atan2(dy, dx))

    if hip_angle >= 45:
        score = 100.0
    elif hip_angle >= 15:
        score = (hip_angle - 15) / 30 * 100
    else:
        score = 0.0

    return score, {"hip_line_angle": hip_angle}


# ---- フォアハンド専用 ----

def _score_weight_shift_forehand(keypoints: list) -> tuple:
    """
    体重移動スコア（フォアハンド用）。
    右足 → 左足への体重移動を評価する。
    腰の重心が左足寄りほど高スコア。
    """
    left_hip    = _get_xy(keypoints, LEFT_HIP)
    right_hip   = _get_xy(keypoints, RIGHT_HIP)
    left_ankle  = _get_xy(keypoints, LEFT_ANKLE)
    right_ankle = _get_xy(keypoints, RIGHT_ANKLE)

    hip_center_x = (left_hip[0] + right_hip[0]) / 2
    left_x  = left_ankle[0]
    right_x = right_ankle[0]

    foot_span = abs(left_x - right_x)
    if foot_span < 1e-6:
        return 50.0, {"hip_center_x": hip_center_x, "weight_ratio": 0.5}

    if left_x < right_x:
        ratio = (hip_center_x - right_x) / (left_x - right_x)
    else:
        ratio = (hip_center_x - left_x) / (right_x - left_x)

    ratio = max(0.0, min(1.0, ratio))
    score = ratio * 100

    return score, {"hip_center_x": hip_center_x, "weight_ratio": ratio}


# ---- バックハンド専用 ----

def _score_weight_shift_backhand(keypoints: list) -> tuple:
    """
    体重移動スコア（バックハンド用）。
    左足 → 右足への体重移動を評価する。
    腰の重心が右足寄りほど高スコア。
    """
    left_hip    = _get_xy(keypoints, LEFT_HIP)
    right_hip   = _get_xy(keypoints, RIGHT_HIP)
    left_ankle  = _get_xy(keypoints, LEFT_ANKLE)
    right_ankle = _get_xy(keypoints, RIGHT_ANKLE)

    hip_center_x = (left_hip[0] + right_hip[0]) / 2
    left_x  = left_ankle[0]
    right_x = right_ankle[0]

    foot_span = abs(left_x - right_x)
    if foot_span < 1e-6:
        return 50.0, {"hip_center_x": hip_center_x, "weight_ratio": 0.5}

    # 右足寄りほど高スコア（フォアハンドの逆）
    if left_x < right_x:
        ratio = (hip_center_x - left_x) / (right_x - left_x)
    else:
        ratio = (hip_center_x - right_x) / (left_x - right_x)

    ratio = max(0.0, min(1.0, ratio))
    score = ratio * 100

    return score, {"hip_center_x": hip_center_x, "weight_ratio": ratio}


# ---- サーブ専用 ----

def _score_body_twist(keypoints: list) -> tuple:
    """
    体の捻りスコア（サーブ用）。

    肩のラインと腰のラインの角度差で捻りを評価する。
    肩と腰の向きがズレているほど（捻れているほど）高スコア。
    """
    left_shoulder  = _get_xy(keypoints, LEFT_SHOULDER)
    right_shoulder = _get_xy(keypoints, RIGHT_SHOULDER)
    left_hip       = _get_xy(keypoints, LEFT_HIP)
    right_hip      = _get_xy(keypoints, RIGHT_HIP)

    # 肩ラインの角度
    shoulder_angle = math.degrees(math.atan2(
        abs(left_shoulder[1] - right_shoulder[1]),
        abs(left_shoulder[0] - right_shoulder[0])
    ))
    # 腰ラインの角度
    hip_angle = math.degrees(math.atan2(
        abs(left_hip[1] - right_hip[1]),
        abs(left_hip[0] - right_hip[0])
    ))

    # 角度差が大きいほど捻れている
    twist = abs(shoulder_angle - hip_angle)

    if twist >= 20:
        score = 100.0
    elif twist >= 5:
        score = (twist - 5) / 15 * 100
    else:
        score = 0.0

    return score, {"shoulder_angle": shoulder_angle, "hip_angle": hip_angle, "twist": twist}


def _score_elbow_height(keypoints: list) -> tuple:
    """
    肘の高さスコア（サーブ用）。

    打つ側の肘（右肘）が肩より上に上がっているか評価する。
    肘y座標 < 肩y座標（画像座標は上が小さい）で高スコア。
    """
    right_shoulder = _get_xy(keypoints, RIGHT_SHOULDER)
    right_elbow    = _get_xy(keypoints, RIGHT_ELBOW)

    # 画像座標は上が小さいため、肘y < 肩y なら肘が上にある
    diff = right_shoulder[1] - right_elbow[1]  # 正なら肘が肩より上

    if diff >= 30:
        score = 100.0
    elif diff >= 0:
        score = diff / 30 * 100
    elif diff >= -30:
        score = max(0.0, 50.0 + diff / 30 * 50)
    else:
        score = 0.0

    return score, {"elbow_above_shoulder_px": diff}


def _score_weight_forward(keypoints: list) -> tuple:
    """
    重心が前（つま先方向）スコア（サーブ用）。

    腰の重心x座標がつま先（足先）より前に出ているか評価する。
    サーブでは体重が前に乗るほど良いフォーム。
    左右の足首の中心より腰が前（画像上で小さいx方向）に出ているほど高スコア。
    """
    left_hip    = _get_xy(keypoints, LEFT_HIP)
    right_hip   = _get_xy(keypoints, RIGHT_HIP)
    left_ankle  = _get_xy(keypoints, LEFT_ANKLE)
    right_ankle = _get_xy(keypoints, RIGHT_ANKLE)

    hip_center_x  = (left_hip[0] + right_hip[0]) / 2
    foot_center_x = (left_ankle[0] + right_ankle[0]) / 2

    # 腰が足より前（x座標が小さい側）に出ているほど高スコア
    diff = foot_center_x - hip_center_x  # 正なら腰が前

    if diff >= 20:
        score = 100.0
    elif diff >= 0:
        score = diff / 20 * 100
    elif diff >= -20:
        score = max(0.0, 50.0 + diff / 20 * 50)
    else:
        score = 0.0

    return score, {"hip_forward_px": diff}


# ---- 公開API ----

def calc_forehand_score(keypoints: list) -> FormScoreResult:
    """フォアハンドのフォームスコアを計算する。"""
    knee_score,   knee_d   = _score_knee_bend(keypoints)
    weight_score, weight_d = _score_weight_shift_forehand(keypoints)
    hip_score,    hip_d    = _score_hip_rotation(keypoints)

    total = (knee_score + weight_score + hip_score) / 3
    details = {**knee_d, **weight_d, **hip_d}

    return FormScoreResult(
        shot="フォアハンド",
        total=round(total, 1),
        scores={"膝の曲がり": round(knee_score, 1), "体重移動": round(weight_score, 1), "腰の回転": round(hip_score, 1)},
        details=details,
    )


def calc_backhand_score(keypoints: list) -> FormScoreResult:
    """バックハンドのフォームスコアを計算する。"""
    knee_score,   knee_d   = _score_knee_bend(keypoints)
    weight_score, weight_d = _score_weight_shift_backhand(keypoints)
    hip_score,    hip_d    = _score_hip_rotation(keypoints)

    total = (knee_score + weight_score + hip_score) / 3
    details = {**knee_d, **weight_d, **hip_d}

    return FormScoreResult(
        shot="バックハンド",
        total=round(total, 1),
        scores={"膝の曲がり": round(knee_score, 1), "体重移動": round(weight_score, 1), "腰の回転": round(hip_score, 1)},
        details=details,
    )


def calc_serve_score(keypoints: list) -> FormScoreResult:
    """サーブのフォームスコアを計算する。"""
    knee_score,    knee_d    = _score_knee_bend(keypoints)
    twist_score,   twist_d   = _score_body_twist(keypoints)
    elbow_score,   elbow_d   = _score_elbow_height(keypoints)
    forward_score, forward_d = _score_weight_forward(keypoints)

    total = (knee_score + twist_score + elbow_score + forward_score) / 4
    details = {**knee_d, **twist_d, **elbow_d, **forward_d}

    return FormScoreResult(
        shot="サーブ",
        total=round(total, 1),
        scores={
            "膝の曲がり": round(knee_score, 1),
            "体の捻り":   round(twist_score, 1),
            "肘の高さ":   round(elbow_score, 1),
            "重心が前":   round(forward_score, 1),
        },
        details=details,
    )


# ショット名からスコア計算関数を引くテーブル
_SCORE_FUNCS = {
    "forehand":       calc_forehand_score,
    "backhand":       calc_backhand_score,
    "serve":          calc_serve_score,
}


def score_from_json(json_path: str, shot: str, sample_idx: int = 0) -> FormScoreResult:
    """
    JSONファイルから直接フォームスコアを計算する。

    Args:
        json_path:  アノテーションJSONのパス
        shot:       ショット種類 ("forehand" / "backhand" / "serve")
        sample_idx: 何番目のサンプルを評価するか

    Returns:
        FormScoreResult
    """
    if shot not in _SCORE_FUNCS:
        raise ValueError(f"shotは 'forehand' / 'backhand' / 'serve' のいずれかを指定してください。got: {shot}")

    with open(json_path, "r") as f:
        data = json.load(f)
    keypoints = data["annotations"][sample_idx]["keypoints"]
    return _SCORE_FUNCS[shot](keypoints)
