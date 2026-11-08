"""탐지: YOLO, 색 마커, 배경 차분(모르는 물건), ROI 저장"""
import os

import cv2
import numpy as np

from config import *
from common import zone_of


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))


def load_yolo():
    """YOLO 엔진 로드 + 워밍업 + 사용할 클래스 이름을 COCO 클래스 ID로 변환 (04-D-6 클래스 지정 탐지)
    yolo.names를 바로 읽으면 엔진이 한 번 더 로드되므로(GPU 메모리 2배), 워밍업 결과에서 클래스 이름을 얻음"""
    from ultralytics import YOLO
    yolo = YOLO(YOLO_MODEL_PATH, task="detect")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    result = yolo.predict(source=dummy, imgsz=YOLO_IMGSZ, verbose=False)[0]   # 첫 추론이 느린 것을 미리 처리
    name_to_id = {name: class_id for class_id, name in result.names.items()}
    class_ids = [name_to_id[name] for name in YOLO_ITEMS + CARRIERS]
    return yolo, class_ids


def detect_yolo(yolo, class_ids, frame):
    """YOLO 탐지 결과를 dict 리스트로 변환 (06-A-2 JSON 구조와 같은 필드)"""
    results = yolo.predict(source=frame, conf=YOLO_CONF, iou=YOLO_IOU, classes=class_ids,
                           imgsz=YOLO_IMGSZ, verbose=False)
    result = results[0]

    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        detections.append({
            "class": result.names[class_id],
            "conf": round(float(box.conf[0].item()), 3),
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
        })
    return detections


def detect_markers(frame):
    """COCO에 없는 물건: 색 스티커를 LAB 변환 → 블러 → 색 분할 → 모폴로지 → 가장 큰 윤곽선의 BBox (Project01)
    LAB의 L(밝기) 범위를 넓게 잡아 두어 밝기 보정(CLAHE) 없이도 조명 변화에 강함"""
    blur = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB), (BLUR_K, BLUR_K), 0)
    detections = []

    for marker_class, marker in MARKERS.items():
        mask = cv2.inRange(blur, marker["low"], marker["high"])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=OPEN_ITER)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITER)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = [c for c in contours if cv2.contourArea(c) > MARKER_MIN_AREA]
        if len(contours) == 0:
            continue

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)        # (x, y, w, h) → (x1, y1, x2, y2) 변환 (04-A)
        detections.append({"class": marker_class, "conf": 1.0, "bbox": (x, y, x + w, y + h)})

    return detections


def to_gray(frame):
    """배경 차분용: 흑백 + 큰 가우시안 블러 (작은 노이즈, 미세한 흔들림 무시)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (DIFF_BLUR, DIFF_BLUR), 0)


def detect_unknown(S, gray, known_bboxes):
    """체크인 배경과 비교해 생긴 변화 중, 이미 아는 물건이 아닌 영역 = 모르는 물건
    배경 차분 → 이진화 → 아는 물건 영역 제거 → 모폴로지 → Contour (02 섹션 기법)"""
    if S["background"] is None:
        return []

    diff = cv2.absdiff(gray, S["background"])
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    for x1, y1, x2, y2 in known_bboxes:
        cv2.rectangle(mask, (x1 - KNOWN_MARGIN, y1 - KNOWN_MARGIN), (x2 + KNOWN_MARGIN, y2 + KNOWN_MARGIN), 0, -1)

    if cv2.countNonZero(mask) > UNKNOWN_MAX_RATIO * mask.size:
        return []                                       # 조명이 바뀌었거나 사람이 화면을 가림

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=OPEN_ITER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITER)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for c in contours:
        if cv2.contourArea(c) < UNKNOWN_MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        bbox = (x, y, x + w, y + h)
        if zone_of(bbox) is not OUTSIDE_ZONE:            # 구역 안에 놓인 것만
            detections.append({"class": UNKNOWN_CLASS, "conf": 0.5, "bbox": bbox})
    return detections


def update_background(S, gray, object_bboxes):
    """물체가 없는 영역만 배경을 천천히 갱신 (햇빛 변화 등으로 생기는 가짜 변화 방지)
    물체가 있는 영역은 갱신하지 않으므로, 새로 놓인 물건이 배경에 묻히지 않음"""
    if S["background"] is None:
        return
    mask = np.full(gray.shape, 255, dtype=np.uint8)
    for x1, y1, x2, y2 in object_bboxes:
        cv2.rectangle(mask, (x1 - KNOWN_MARGIN, y1 - KNOWN_MARGIN), (x2 + KNOWN_MARGIN, y2 + KNOWN_MARGIN), 0, -1)
    background = S["background"].astype(np.float32)
    cv2.accumulateWeighted(gray.astype(np.float32), background, BG_ALPHA, mask=mask)
    S["background"] = background.astype(np.uint8)


def save_roi(frame, bbox, path):
    """BBox 영역을 잘라 저장 (06-B ROI 저장 방식: 임시 파일 → os.replace)"""
    h, w = frame.shape[:2]
    x1, y1 = max(0, bbox[0] - ROI_PAD), max(0, bbox[1] - ROI_PAD)
    x2, y2 = min(w, bbox[2] + ROI_PAD), min(h, bbox[3] + ROI_PAD)
    roi = frame[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path.replace(".jpg", "_tmp.jpg")
    if not cv2.imwrite(tmp_path, roi):
        return False
    os.replace(tmp_path, path)
    return True
