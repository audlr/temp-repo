import json
import os
import time

import numpy as np

from config import ITEM_NAMES_KO


def write_json(path, data):
    """임시 파일에 먼저 쓴 뒤 os.replace로 교체 (06-A-7 방식, 읽는 쪽이 쓰다 만 파일을 읽지 않음)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    os.replace(tmp_path, path)


def read_json(path, default=None):
    """파일이 없으면 default 반환 (06-A-7처럼 os.path.exists로 확인. 쓰는 쪽이 os.replace를 써서 깨진 파일은 읽지 않음)"""
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


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
    (예: 같은 가방을 handbag과 backpack으로 동시에 잡은 경우, YOLO와 색 마커가 같은 물체를 잡은 경우)"""
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


# 체크아웃 판단 기준: vision·assistant가 같은 기준을 쓰도록 여기 한 곳에서 정의
# item은 vision의 track이든 state.json의 물건 기록이든 같은 키(class, status, owner)를 가짐
def is_left(item, exceptions):
    """두고 가면 안 되는 물건: 체크인 후 생긴 내 물건 + 지금 있음 + 두고 가도 된다고 하지 않음"""
    return item["status"] == "present" and item["owner"] == "mine" and item["class"] not in exceptions


def is_missing_amenity(item):
    """체크인 때 있던 숙소 비품이 사라짐 (짐에 같이 넣었을 수 있음)"""
    return item["status"] in ("missing", "taken") and item["owner"] == "amenity"


def name_ko(item_class):
    return ITEM_NAMES_KO.get(item_class, item_class)


def josa(word, pair):
    """받침 유무에 따라 조사 선택. pair = ("이", "가"), ("은", "는"), ("을", "를"), ("으로", "로")"""
    last = word[-1]
    if last.isdigit():                                   # 숫자는 읽는 소리로 받침 판단 (1 일, 2 이, 3 삼 ...)
        last = "영일이삼사오육칠팔구"[int(last)]
    if "가" <= last <= "힣":
        final = (ord(last) - ord("가")) % 28
        if pair == ("으로", "로") and final == 8:        # ㄹ 받침은 "로" (예: 물병로 X, 칼로 O)
            return word + "로"
        return word + (pair[0] if final else pair[1])
    return word + pair[1]


def ago_ko(timestamp, now=None):
    """경과 시간을 '방금', '3분 전', '1시간 전'으로 표현"""
    sec = (now or time.time()) - timestamp
    if sec < 60:
        return "방금"
    if sec < 3600:
        return f"{int(sec // 60)}분 전"
    return f"{int(sec // 3600)}시간 전"
