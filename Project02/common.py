import json
import os
import time

import numpy as np

from config import ITEM_NAMES_KO, CARRIERS, ZONES, OUTSIDE_ZONE


def cleanup_old_files(folders, keep_sec, now):
    """keep_sec이 지난 파일 삭제 (부재 사진, 모르는 물건 ROI를 기기에 오래 남기지 않기 위해)"""
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for file_name in os.listdir(folder):
            path = os.path.join(folder, file_name)
            if now - os.path.getmtime(path) > keep_sec:
                os.remove(path)
                print("[자동 삭제]", path)


def write_json(path, data):
    """임시 파일에 먼저 쓴 뒤 os.replace로 교체 (06-A-7 방식, 읽는 쪽이 쓰다 만 파일을 읽지 않음)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    os.replace(tmp_path, path)


def read_json(path, default=None):
    """파일이 없거나 읽는 도중 깨져 있으면 default 반환"""
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def calculate_iou(bbox_a, bbox_b):
    """(x1, y1, x2, y2) 두 박스의 IoU"""
    xa_1, ya_1, xa_2, ya_2 = bbox_a
    xb_1, yb_1, xb_2, yb_2 = bbox_b

    w = max(0, min(xa_2, xb_2) - max(xa_1, xb_1))
    h = max(0, min(ya_2, yb_2) - max(ya_1, yb_1))

    intersection_area = w * h
    union_area = (xa_2 - xa_1) * (ya_2 - ya_1) + (xb_2 - xb_1) * (yb_2 - yb_1) - intersection_area

    if union_area <= 0:
        return 0.0
    return intersection_area / union_area


def apply_nms(detections, iou_threshold):
    """클래스와 무관하게 겹치는 탐지 중 confidence가 가장 높은 것만 남김
    (예: 같은 가방을 handbag과 backpack으로 동시에 잡은 경우)"""
    if len(detections) == 0:
        return []

    order = list(np.argsort([d["conf"] for d in detections])[::-1])
    keep = []

    while len(order) > 0:
        best = order[0]
        keep.append(detections[best])
        order = [i for i in order[1:]
                 if calculate_iou(detections[best]["bbox"], detections[i]["bbox"]) <= iou_threshold]

    return keep


def zone_of(bbox):
    """BBox 중심점이 들어 있는 구역"""
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    for zone in ZONES:
        x1, y1, x2, y2 = zone["rect"]
        if x1 <= cx < x2 and y1 <= cy < y2:
            return zone
    return OUTSIDE_ZONE


# 남은 물건 판단 기준: vision·assistant가 같은 기준을 쓰도록 여기 한 곳에서 정의
# item은 vision의 track이든 state.json의 물건 기록이든 같은 키(id, class, status, owner)를 가짐
def is_exception(item, exceptions):
    """두고 가도 되는 물건: 클래스로 지정("umbrella") 또는 이름 붙인 물건을 id로 지정("id:3")"""
    return item["class"] in exceptions or f"id:{item['id']}" in exceptions


def is_my_item(item):
    """내 소지품: 체크인 후 생긴 물건 (비품·무시 영역·들고 나가는 가방 제외)"""
    return item["owner"] == "mine" and item["class"] not in CARRIERS


def is_left(item, exceptions):
    """출발할 때 남아 있으면 안 되는 물건: 내 소지품 + 지금 있음 + 예외 아님"""
    return item["status"] == "present" and is_my_item(item) and not is_exception(item, exceptions)


def name_ko(item_class):
    return ITEM_NAMES_KO.get(item_class, item_class)


def track_name(t):
    """vision 기록의 한국어 이름: Gemma가 붙인 이름이 있으면 그 이름, 없으면 클래스 이름"""
    return t["label"] or name_ko(t["class"])


def josa(word, pair):
    """받침 유무에 따라 조사 선택. pair = ("이", "가"), ("은", "는"), ("을", "를")"""
    last = word[-1]
    if "가" <= last <= "힣":
        has_final = (ord(last) - ord("가")) % 28 != 0
        return word + (pair[0] if has_final else pair[1])
    return word + pair[1]


def clock_ko(timestamp):
    """1731050000.0 → '14시 32분'"""
    t = time.localtime(timestamp)
    return f"{t.tm_hour}시 {t.tm_min}분"


def ago_ko(timestamp, now=None):
    """경과 시간을 '방금', '3분 전', '1시간 전'으로 표현"""
    sec = (now or time.time()) - timestamp
    if sec < 60:
        return "방금"
    if sec < 3600:
        return f"{int(sec // 60)}분 전"
    return f"{int(sec // 3600)}시간 전"
