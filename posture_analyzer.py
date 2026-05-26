"""
Posture analysis from a photo.

This script is intended for an educational digital project. It uses
YOLO Pose to find human body landmarks, then calculates simple posture
metrics: shoulder tilt, hip tilt, head offset, and torso lean.

The result is not a medical diagnosis. It is a demonstration of how neural
network landmarks can be used for a practical computer vision task.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LANDMARK_NAMES = [
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
]

POSE_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
]

YOLO_LANDMARK_NAMES = [
    "NOSE",
    "LEFT_EYE",
    "RIGHT_EYE",
    "LEFT_EAR",
    "RIGHT_EAR",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
]

YOLO_CONNECTIONS = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    z: float
    visibility: float


@dataclass(frozen=True)
class Metric:
    name: str
    value: float
    unit: str
    status: str
    comment: str
    penalty: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 2),
            "unit": self.unit,
            "status": self.status,
            "comment": self.comment,
        }


def prepare_runtime_dirs() -> None:
    matplotlib_cache = Path(".cache/matplotlib").resolve()
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    yolo_cache = Path(".cache/ultralytics").resolve()
    yolo_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_cache))


def import_cv2():
    try:
        prepare_runtime_dirs()
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "Не найдены нужные библиотеки. Установите их командой:\n"
            "pip install -r requirements.txt"
        ) from exc

    return cv2


def import_yolo():
    try:
        prepare_runtime_dirs()
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Не найдена библиотека ultralytics. Установите зависимости командой:\n"
            "pip install -r requirements.txt"
        ) from exc

    return YOLO


def import_mediapipe():
    try:
        prepare_runtime_dirs()
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit(
            "MediaPipe не установлен. Используйте обычный запуск через YOLO "
            "или установите MediaPipe отдельно."
        ) from exc

    if not hasattr(mp, "solutions") and not hasattr(mp, "tasks"):
        raise SystemExit(
            "Backend MediaPipe сейчас недоступен: установлен пустой или несовместимый пакет mediapipe.\n"
            "Для этого проекта используйте основной рабочий вариант:\n"
            "python posture_analyzer.py bad21.jpg\n\n"
            "Параметр --backend mediapipe оставлен только как экспериментальный запасной режим."
        )

    return cv2, mp


def midpoint(a: Point, b: Point) -> Point:
    return Point(
        x=(a.x + b.x) / 2,
        y=(a.y + b.y) / 2,
        z=(a.z + b.z) / 2,
        visibility=min(a.visibility, b.visibility),
    )


def distance(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def angle_to_horizontal(a: Point, b: Point) -> float:
    angle = abs(math.degrees(math.atan2(b.y - a.y, b.x - a.x)))
    if angle > 90:
        angle = 180 - angle
    return angle


def angle_to_vertical(a: Point, b: Point) -> float:
    return abs(90 - angle_to_horizontal(a, b))


def percent(part: float, whole: float) -> float:
    if whole <= 0:
        return 0.0
    return part / whole * 100


def classify_metric(
    name: str,
    value: float,
    unit: str,
    good_limit: float,
    warning_limit: float,
    good_comment: str,
    warning_comment: str,
    bad_comment: str,
) -> Metric:
    if value <= good_limit:
        return Metric(name, value, unit, "норма", good_comment, 0)
    if value <= warning_limit:
        return Metric(name, value, unit, "есть отклонение", warning_comment, 8)
    return Metric(name, value, unit, "выраженное отклонение", bad_comment, 18)


def landmark_visibility(landmark: Any) -> float:
    visibility = getattr(landmark, "visibility", None)
    if visibility is None:
        visibility = getattr(landmark, "presence", None)
    if visibility is None:
        return 1.0
    return float(visibility)


def landmarks_to_points(landmarks: list[Any], min_visibility: float) -> dict[str, Point]:
    points: dict[str, Point] = {}
    for index, name in enumerate(LANDMARK_NAMES):
        lm = landmarks[index]
        visibility = landmark_visibility(lm)
        if visibility >= min_visibility:
            points[name] = Point(lm.x, lm.y, getattr(lm, "z", 0.0), visibility)
    return points


def yolo_landmarks_to_points(
    landmarks: list[Point], min_visibility: float
) -> dict[str, Point]:
    points: dict[str, Point] = {}
    for index, name in enumerate(YOLO_LANDMARK_NAMES):
        point = landmarks[index]
        if point.visibility >= min_visibility:
            points[name] = point
    return points


def find_yolo_model_path(model_path: Path | None) -> str:
    if model_path is not None:
        if model_path.exists():
            return str(model_path)
        raise SystemExit(f"Файл модели не найден: {model_path}")

    for candidate in [
        Path("models/yolo11n-pose.pt"),
        Path("yolo11n-pose.pt"),
        Path("models/yolov8n-pose.pt"),
        Path("yolov8n-pose.pt"),
    ]:
        if candidate.exists():
            return str(candidate)

    return "yolo11n-pose.pt"


def find_model_path(model_path: Path | None) -> Path:
    candidates = []
    if model_path is not None:
        candidates.append(model_path)

    candidates.extend(
        [
            Path("models/pose_landmarker_full.task"),
            Path("pose_landmarker_full.task"),
            Path("pose_landmarker.task"),
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise SystemExit(
        "Для установленной версии mediapipe нужен файл модели Pose Landmarker.\n"
        "Скачайте его в папку models под именем pose_landmarker_full.task:\n"
        "curl -L -o models/pose_landmarker_full.task "
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task\n"
        "После этого снова запустите программу."
    )


def extract_yolo_pose_points(
    image, min_visibility: float = 0.5, model_path: Path | None = None
) -> tuple[dict[str, Point], list[Point], list[tuple[int, int]]]:
    YOLO = import_yolo()
    model = YOLO(find_yolo_model_path(model_path))
    results = model.predict(
        source=image,
        imgsz=960,
        conf=0.25,
        device="cpu",
        verbose=False,
    )

    if not results or results[0].keypoints is None:
        raise SystemExit(
            "На фото не удалось найти человека. "
            "Попробуйте фото в полный рост, с хорошим светом и без сильного поворота тела."
        )

    result = results[0]
    keypoints = result.keypoints
    xy = keypoints.xyn.cpu().numpy()
    if xy.shape[0] == 0:
        raise SystemExit(
            "На фото не удалось найти ключевые точки тела. "
            "Попробуйте фото, где человек виден полностью."
        )

    if keypoints.conf is None:
        confidence = [[1.0] * len(YOLO_LANDMARK_NAMES) for _ in range(xy.shape[0])]
    else:
        confidence = keypoints.conf.cpu().numpy()

    person_index = 0
    if result.boxes is not None and result.boxes.xyxy is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        if len(boxes) > 0:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            person_index = int(areas.argmax())

    landmarks = [
        Point(
            x=float(xy[person_index][index][0]),
            y=float(xy[person_index][index][1]),
            z=0.0,
            visibility=float(confidence[person_index][index]),
        )
        for index in range(len(YOLO_LANDMARK_NAMES))
    ]
    points = yolo_landmarks_to_points(landmarks, min_visibility)
    return points, landmarks, YOLO_CONNECTIONS


def extract_mediapipe_pose_points(
    image, min_visibility: float = 0.5, model_path: Path | None = None
) -> tuple[dict[str, Point], list[Any], list[tuple[int, int]]]:
    cv2, mp = import_mediapipe()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if hasattr(mp, "solutions"):
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        ) as pose:
            result = pose.process(rgb)

        if not result.pose_landmarks:
            raise SystemExit(
                "На фото не удалось найти человека. "
                "Попробуйте фото в полный рост, с хорошим светом и без сильного поворота тела."
            )

        landmarks = list(result.pose_landmarks.landmark)
        points = landmarks_to_points(landmarks, min_visibility)
        return points, landmarks, POSE_CONNECTIONS

    model = find_model_path(model_path)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        output_segmentation_masks=False,
    )

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        raise SystemExit(
            "На фото не удалось найти человека. "
            "Попробуйте фото в полный рост, с хорошим светом и без сильного поворота тела."
        )

    landmarks = list(result.pose_landmarks[0])
    points = landmarks_to_points(landmarks, min_visibility)
    return points, landmarks, POSE_CONNECTIONS


def extract_pose_points(
    image,
    min_visibility: float = 0.25,
    model_path: Path | None = None,
    backend: str = "yolo",
) -> tuple[dict[str, Point], list[Any], list[tuple[int, int]]]:
    if backend == "mediapipe":
        return extract_mediapipe_pose_points(image, min_visibility, model_path)
    return extract_yolo_pose_points(image, min_visibility, model_path)


def require_points(points: dict[str, Point], names: list[str]) -> list[Point]:
    missing = [name for name in names if name not in points]
    if missing:
        raise SystemExit(
            "Не хватает ключевых точек тела: "
            + ", ".join(missing)
            + ". Попробуйте фото, где человек виден полностью."
        )
    return [points[name] for name in names]


def head_center(points: dict[str, Point]) -> Point:
    if "NOSE" in points:
        return points["NOSE"]
    if "LEFT_EAR" in points and "RIGHT_EAR" in points:
        return midpoint(points["LEFT_EAR"], points["RIGHT_EAR"])
    raise SystemExit("Не удалось определить положение головы на фото.")


def detect_view(points: dict[str, Point]) -> str:
    left_shoulder, right_shoulder, left_hip, right_hip = require_points(
        points, ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]
    )
    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    hip_mid = midpoint(left_hip, right_hip)
    torso_height = max(abs(shoulder_mid.y - hip_mid.y), 0.001)
    shoulder_ratio = distance(left_shoulder, right_shoulder) / torso_height
    return "side" if shoulder_ratio < 0.45 else "front"


def analyze_front_posture(points: dict[str, Point]) -> dict[str, Any]:
    left_shoulder, right_shoulder, left_hip, right_hip = require_points(
        points, ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]
    )

    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    hip_mid = midpoint(left_hip, right_hip)
    shoulder_width = distance(left_shoulder, right_shoulder)
    head = head_center(points)

    shoulder_tilt = angle_to_horizontal(left_shoulder, right_shoulder)
    hip_tilt = angle_to_horizontal(left_hip, right_hip)
    head_offset = percent(abs(head.x - shoulder_mid.x), shoulder_width)
    torso_lean = angle_to_vertical(hip_mid, shoulder_mid)

    metrics = [
        classify_metric(
            "Наклон плеч",
            shoulder_tilt,
            "град.",
            3,
            7,
            "Линия плеч почти горизонтальна.",
            "Плечи немного находятся на разной высоте.",
            "Плечи заметно находятся на разной высоте.",
        ),
        classify_metric(
            "Наклон таза",
            hip_tilt,
            "град.",
            4,
            8,
            "Линия таза почти горизонтальна.",
            "Есть небольшая асимметрия таза.",
            "Есть заметная асимметрия таза.",
        ),
        classify_metric(
            "Смещение головы от центра плеч",
            head_offset,
            "%",
            8,
            15,
            "Голова расположена близко к центральной линии плеч.",
            "Голова немного смещена относительно центра плеч.",
            "Голова заметно смещена относительно центра плеч.",
        ),
        classify_metric(
            "Наклон корпуса",
            torso_lean,
            "град.",
            5,
            10,
            "Корпус расположен близко к вертикали.",
            "Корпус немного отклонен от вертикали.",
            "Корпус заметно отклонен от вертикали.",
        ),
    ]

    score = max(0, 100 - sum(metric.penalty for metric in metrics))
    recommendations = build_recommendations(metrics)

    return {
        "view": "фото спереди",
        "score": score,
        "metrics": metrics,
        "recommendations": recommendations,
        "landmarks": {
            "left_shoulder": left_shoulder,
            "right_shoulder": right_shoulder,
            "left_hip": left_hip,
            "right_hip": right_hip,
            "shoulder_mid": shoulder_mid,
            "hip_mid": hip_mid,
            "head": head,
        },
    }


def side_score(points: dict[str, Point], prefix: str) -> float:
    names = [
        f"{prefix}_EAR",
        f"{prefix}_SHOULDER",
        f"{prefix}_HIP",
        f"{prefix}_KNEE",
        f"{prefix}_ANKLE",
    ]
    return sum(points[name].visibility for name in names if name in points)


def choose_side_points(points: dict[str, Point]) -> tuple[str, Point, Point, Point]:
    candidates = []
    for prefix in ["LEFT", "RIGHT"]:
        shoulder_name = f"{prefix}_SHOULDER"
        hip_name = f"{prefix}_HIP"
        if shoulder_name in points and hip_name in points:
            candidates.append((side_score(points, prefix), prefix))

    if not candidates:
        raise SystemExit(
            "Не хватает ключевых точек плеча и таза. "
            "Попробуйте фото, где человек виден от головы до бедер."
        )

    _, prefix = max(candidates)
    opposite = "RIGHT" if prefix == "LEFT" else "LEFT"

    head = None
    for name in [f"{prefix}_EAR", f"{opposite}_EAR", "NOSE"]:
        if name in points:
            head = points[name]
            break
    if head is None:
        raise SystemExit("Не удалось определить положение головы на фото.")

    return prefix, head, points[f"{prefix}_SHOULDER"], points[f"{prefix}_HIP"]


def analyze_side_posture(points: dict[str, Point]) -> dict[str, Any]:
    prefix, head, shoulder, hip = choose_side_points(points)
    torso_length = max(distance(shoulder, hip), 0.001)

    head_forward = percent(abs(head.x - shoulder.x), torso_length)
    neck_lean = angle_to_vertical(shoulder, head)
    torso_lean = angle_to_vertical(hip, shoulder)
    shoulder_hip_offset = percent(abs(shoulder.x - hip.x), torso_length)

    metrics = [
        classify_metric(
            "Выдвижение головы вперед",
            head_forward,
            "%",
            8,
            18,
            "Голова расположена близко к вертикали плеча.",
            "Голова немного вынесена вперед относительно плеча.",
            "Голова заметно вынесена вперед относительно плеча.",
        ),
        classify_metric(
            "Наклон шеи",
            neck_lean,
            "град.",
            10,
            20,
            "Линия от плеча к голове близка к вертикали.",
            "Шея немного наклонена вперед.",
            "Шея заметно наклонена вперед.",
        ),
        classify_metric(
            "Наклон корпуса",
            torso_lean,
            "град.",
            5,
            10,
            "Корпус расположен близко к вертикали.",
            "Корпус немного отклонен от вертикали.",
            "Корпус заметно отклонен от вертикали.",
        ),
        classify_metric(
            "Смещение плеча относительно таза",
            shoulder_hip_offset,
            "%",
            10,
            20,
            "Плечо расположено близко к вертикали таза.",
            "Плечо немного смещено относительно таза.",
            "Плечо заметно смещено относительно таза.",
        ),
    ]

    score = max(0, 100 - sum(metric.penalty for metric in metrics))
    recommendations = build_recommendations(metrics)
    side_name = "левая сторона" if prefix == "LEFT" else "правая сторона"

    return {
        "view": f"фото сбоку ({side_name})",
        "score": score,
        "metrics": metrics,
        "recommendations": recommendations,
        "landmarks": {
            "head": head,
            "side_shoulder": shoulder,
            "side_hip": hip,
        },
    }


def analyze_posture(points: dict[str, Point]) -> dict[str, Any]:
    if detect_view(points) == "side":
        return analyze_side_posture(points)
    return analyze_front_posture(points)


def build_recommendations(metrics: list[Metric]) -> list[str]:
    recommendations: list[str] = []

    for metric in metrics:
        if metric.status == "норма":
            continue
        if metric.name == "Наклон плеч":
            recommendations.append(
                "Проверить симметрию плеч: повторить фото без сумки, с расслабленными руками и ровной постановкой ног."
            )
        elif metric.name == "Наклон таза":
            recommendations.append(
                "Проверить положение таза и опоры на ноги: сделать повторное фото на ровной поверхности."
            )
        elif metric.name == "Смещение головы от центра плеч":
            recommendations.append(
                "Проверить положение головы: камера должна быть строго напротив человека, без наклона."
            )
        elif metric.name == "Наклон корпуса":
            recommendations.append(
                "Проверить вертикальность корпуса: повторить съемку в полный рост и сравнить результат."
            )
        elif metric.name == "Выдвижение головы вперед":
            recommendations.append(
                "Сделать контрольное боковое фото: ухо, плечо и таз должны быть хорошо видны."
            )
        elif metric.name == "Наклон шеи":
            recommendations.append(
                "Проверить положение головы и шеи: повторить фото без наклона камеры и сравнить линию ухо-плечо."
            )
        elif metric.name == "Смещение плеча относительно таза":
            recommendations.append(
                "Проверить положение плеча относительно таза на боковом фото с ровной постановкой ног."
            )

    if not recommendations:
        recommendations.append(
            "Серьезных отклонений по выбранным признакам не обнаружено. Для проекта можно сравнить результат с другими фото."
        )

    recommendations.append(
        "Результат является учебной оценкой по изображению и не заменяет консультацию врача или специалиста."
    )
    return recommendations


def point_to_pixel(point: Point, width: int, height: int) -> tuple[int, int]:
    return int(point.x * width), int(point.y * height)


def draw_pose_skeleton(
    image, pose_landmarks: list[Any], connections: list[tuple[int, int]]
) -> None:
    cv2 = import_cv2()
    height, width = image.shape[:2]

    def visible(index: int) -> bool:
        if index >= len(pose_landmarks):
            return False
        landmark = pose_landmarks[index]
        return landmark_visibility(landmark) >= 0.25

    def pixel(index: int) -> tuple[int, int]:
        landmark = pose_landmarks[index]
        return int(landmark.x * width), int(landmark.y * height)

    for start, end in connections:
        if visible(start) and visible(end):
            cv2.line(image, pixel(start), pixel(end), (180, 220, 80), 2)

    for index, landmark in enumerate(pose_landmarks):
        if not visible(index):
            continue
        x, y = pixel(index)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(image, (x, y), 4, (40, 120, 255), -1)
            cv2.circle(image, (x, y), 5, (255, 255, 255), 1)


def draw_analysis(
    image,
    pose_landmarks: list[Any],
    connections: list[tuple[int, int]],
    analysis: dict[str, Any],
):
    cv2 = import_cv2()
    output = image.copy()
    height, width = output.shape[:2]
    landmarks = analysis["landmarks"]

    draw_pose_skeleton(output, pose_landmarks, connections)

    def line(a_name: str, b_name: str, color: tuple[int, int, int], thickness: int = 3) -> None:
        a = point_to_pixel(landmarks[a_name], width, height)
        b = point_to_pixel(landmarks[b_name], width, height)
        cv2.line(output, a, b, color, thickness)

    if "side_shoulder" in landmarks:
        line("head", "side_shoulder", (255, 80, 80), 3)
        line("side_shoulder", "side_hip", (60, 255, 60), 3)
    else:
        line("left_shoulder", "right_shoulder", (0, 220, 255))
        line("left_hip", "right_hip", (0, 160, 255))
        line("shoulder_mid", "hip_mid", (60, 255, 60))
        line("head", "shoulder_mid", (255, 80, 80), 2)

    cv2.putText(
        output,
        f"Posture score: {analysis['score']}/100",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (30, 30, 30),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"Posture score: {analysis['score']}/100",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    image_labels = {
        "Наклон плеч": "Shoulder tilt",
        "Наклон таза": "Hip tilt",
        "Смещение головы от центра плеч": "Head offset",
        "Наклон корпуса": "Torso lean",
        "Выдвижение головы вперед": "Forward head",
        "Наклон шеи": "Neck lean",
        "Смещение плеча относительно таза": "Shoulder offset",
    }

    y = 82
    for metric in analysis["metrics"]:
        label = f"{image_labels[metric.name]}: {metric.value:.1f}"
        cv2.putText(
            output,
            label,
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (30, 30, 30),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            label,
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 30

    return output


def save_text_report(report_path: Path, source_path: Path, analysis: dict[str, Any]) -> None:
    lines = [
        "Анализ осанки человека по фото",
        "",
        f"Файл: {source_path}",
        f"Тип снимка: {analysis['view']}",
        f"Итоговая оценка: {analysis['score']}/100",
        "",
        "Показатели:",
    ]

    for metric in analysis["metrics"]:
        lines.append(
            f"- {metric.name}: {metric.value:.2f} {metric.unit} "
            f"({metric.status}). {metric.comment}"
        )

    lines.extend(["", "Рекомендации:"])
    for recommendation in analysis["recommendations"]:
        lines.append(f"- {recommendation}")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def save_json_report(json_path: Path, source_path: Path, analysis: dict[str, Any]) -> None:
    data = {
        "source": str(source_path),
        "view": analysis["view"],
        "score": analysis["score"],
        "metrics": [metric.to_dict() for metric in analysis["metrics"]],
        "recommendations": analysis["recommendations"],
    }
    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Анализ осанки человека по фото с помощью YOLO Pose."
    )
    parser.add_argument("image", type=Path, help="Путь к фотографии человека.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Папка для сохранения результата.",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.25,
        help="Минимальная уверенность видимости ключевой точки от 0 до 1.",
    )
    parser.add_argument(
        "--backend",
        choices=["yolo", "mediapipe"],
        default="yolo",
        help="Нейросетевая модель для поиска точек тела. Основной рабочий вариант: yolo.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Путь к файлу модели. Для YOLO можно указать .pt, для MediaPipe - .task.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cv2 = import_cv2()

    if not args.image.exists():
        raise SystemExit(f"Файл не найден: {args.image}")

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit("Не удалось открыть изображение. Используйте JPG или PNG.")

    points, pose_landmarks, connections = extract_pose_points(
        image, args.min_visibility, args.model, args.backend
    )
    analysis = analyze_posture(points)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    annotated_path = args.output_dir / f"{stem}_posture.png"
    report_path = args.output_dir / f"{stem}_report.txt"
    json_path = args.output_dir / f"{stem}_report.json"

    annotated = draw_analysis(image, pose_landmarks, connections, analysis)
    cv2.imwrite(str(annotated_path), annotated)
    save_text_report(report_path, args.image, analysis)
    save_json_report(json_path, args.image, analysis)

    print(f"Готово. Итоговая оценка: {analysis['score']}/100")
    print(f"Размеченное фото: {annotated_path}")
    print(f"Текстовый отчет: {report_path}")
    print(f"JSON-отчет: {json_path}")


if __name__ == "__main__":
    main()
