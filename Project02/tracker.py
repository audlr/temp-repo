"""추적: 사진 사이 같은 물체 연결, 있음/미탐지 판정, 변화 기록, 위치 설명

track 상태: candidate(확인 중) → present(있음) ↔ missing(화면에서 사라짐)
missing은 실제로 챙겼다는 뜻이 아니며, 다시 탐지되면 present로 복귀한다.
owner: unknown(체크인 전) / amenity(체크인 때 있던 숙소 비품) / mine(체크인 후 생긴 내 물건)
"""
import math
from collections import deque

from config import *
from common import calculate_iou, name_ko


def new_track(S, det, now):
    S["next_id"] += 1
    return {"id": S["next_id"], "class": det["class"], "bbox": det["bbox"],
            "hits": deque([1], maxlen=STABLE_WINDOW), "status": "candidate",
            "owner": "mine" if S["mode"] == "WATCH" else "unknown",
            "where": "", "anchor": center(det["bbox"]), "anchor_bbox": det["bbox"],
            "first_seen": now, "last_seen": now, "taken_at": None, "missing_since": None}


def match_detections(tracks, detections):
    """같은 클래스끼리 IoU가 가장 큰 쌍부터 매칭 (Greedy). (track, det) 쌍과 남은 탐지 반환"""
    pairs = []
    for ti, t in enumerate(tracks):
        for di, d in enumerate(detections):
            if t["class"] == d["class"]:
                iou = calculate_iou(t["bbox"], d["bbox"])
                if iou >= MATCH_IOU:
                    pairs.append((iou, ti, di))

    matched, used_t, used_d = [], set(), set()
    for iou, ti, di in sorted(pairs, reverse=True):
        if ti not in used_t and di not in used_d:
            matched.append((tracks[ti], detections[di]))
            used_t.add(ti)
            used_d.add(di)

    unmatched = [d for di, d in enumerate(detections) if di not in used_d]
    return matched, unmatched


def confirm(S, t):
    """candidate가 안정적으로 보이면 present로 확정
    같은 종류인데 사라졌거나 이번 사진에서 안 보인 기록이 있으면 그 기록을 되살림
    (물건을 멀리 옮긴 경우: IoU 매칭이 끊겨도 같은 물건으로 보고 '이동'으로 기록, 비품·내 물건 구분도 유지)"""
    gone = [o for o in S["tracks"] if o is not t and o["class"] == t["class"]
            and (o["status"] in ("missing", "taken") or (o["status"] == "present" and o["last_seen"] < t["last_seen"]))]
    if gone:
        old = max(gone, key=lambda o: o["last_seen"])
        old.update(bbox=t["bbox"], hits=t["hits"], status="present", last_seen=t["last_seen"],
                   anchor=center(t["bbox"]), anchor_bbox=t["bbox"], taken_at=None, missing_since=None)
        S["tracks"].remove(t)
        return old, "moved"
    t["status"] = "present"
    return t, "appeared"


def update_tracks(S, detections, now):
    """탐지 결과로 track 갱신. 이번 사진에서 생긴 변화 [(종류, track)] 반환 (appeared / moved / taken)"""
    matched, unmatched = match_detections(S["tracks"], detections)
    seen = set()
    changes = []

    for t, d in matched:
        previous_status = t["status"]
        t.update(bbox=d["bbox"], last_seen=now)
        t["hits"].append(1)
        seen.add(t["id"])

        if previous_status == "missing":
            t["status"], t["missing_since"] = "present", None
            t["taken_at"] = None
            changes.append(("returned", t))
        elif t["status"] == "present":
            reference_bbox = t.get("anchor_bbox", t["bbox"])
            reference_size = math.hypot(reference_bbox[2] - reference_bbox[0],
                                        reference_bbox[3] - reference_bbox[1])
            reference_size = max(reference_size, 1.0)
            relative_move = math.dist(center(d["bbox"]), t["anchor"]) / reference_size
            if relative_move > MOVE_RATIO:
                t["anchor"] = center(d["bbox"])
                t["anchor_bbox"] = d["bbox"]
                changes.append(("moved", t))

    for t in list(S["tracks"]):
        if t["id"] not in seen:
            t["hits"].append(0)

        if t["status"] == "candidate":
            if sum(t["hits"]) >= STABLE_HITS:
                confirmed, kind = confirm(S, t)
                confirmed["anchor"] = center(confirmed["bbox"])
                changes.append((kind, confirmed))
            elif now - t["last_seen"] > MISSING_SEC:
                S["tracks"].remove(t)                  # 잠깐 보였던 오탐지 제거

        elif t["status"] == "present" and now - t["last_seen"] > MISSING_SEC:
            # 화면에서 일정 시간 보이지 않는다는 사실만 기록한다.
            # 카메라 사각지대/가림/탐지 실패일 수 있으므로 '챙김'으로 단정하지 않는다.
            t["status"], t["missing_since"] = "missing", now
            changes.append(("missing", t))

    for d in unmatched:
        S["tracks"].append(new_track(S, d, now))

    return changes


def summarize(t):
    return {"id": t["id"], "class": t["class"], "name": name_ko(t["class"]),
            "where": t["where"], "status": t["status"], "owner": t["owner"],
            "first_seen": t["first_seen"], "last_seen": t["last_seen"],
            "taken_at": t["taken_at"], "missing_since": t.get("missing_since")}


def center(bbox):
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def overlap_ratio(bbox, ref):
    """bbox 면적 중 ref와 겹치는 비율"""
    w = max(0, min(bbox[2], ref[2]) - max(bbox[0], ref[0]))
    h = max(0, min(bbox[3], ref[3]) - max(bbox[1], ref[1]))
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    return w * h / area if area > 0 else 0.0


def gap(a, b):
    """두 BBox 테두리 사이 거리 (겹치면 0)"""
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy)


def box_size(bbox):
    return max(bbox[2] - bbox[0], bbox[3] - bbox[1], 1)


def screen_position(bbox, shape):
    """주변에 기준 물건이 없을 때: 화면을 3×3으로 나눈 위치 ("화면 왼쪽 위", "화면 가운데")"""
    h, w = shape[:2]
    cx, cy = center(bbox)
    col = "왼쪽" if cx < w / 3 else "오른쪽" if cx > 2 * w / 3 else "가운데"
    row = "위" if cy < h / 3 else "아래" if cy > 2 * h / 3 else ""
    return f"화면 {col} {row}".strip()


def describe_location(t, landmarks, shape):
    """주변 물건 기준 위치: '노트북 위', '노트북 왼쪽', '컵 아래쪽', 가까운 물건이 없으면 '화면 왼쪽 위'
    기준 물건은 테두리 사이 간격이 가장 좁은 물건 (중심 거리로 고르면 큰 물건이 늘 기준이 되어 부정확)"""
    cx, cy = center(t["bbox"])
    candidates = []
    for o in landmarks:
        if o is t or overlap_ratio(o["bbox"], t["bbox"]) >= ON_TOP_RATIO:
            continue                                  # 자기 위에 놓인 작은 물건은 기준으로 쓰지 않음
        if box_size(o["bbox"]) < MIN_REF_RATIO * box_size(t["bbox"]):
            continue                                  # 훨씬 작은 물건 기준은 어색함 ("물병 왼쪽에 노트북")
        rx, ry = center(o["bbox"])
        distance = gap(t["bbox"], o["bbox"])
        if overlap_ratio(t["bbox"], o["bbox"]) >= ON_TOP_RATIO:
            candidates.append((-1.0, o, "위"))
        elif distance <= NEAR_RATIO * max(box_size(t["bbox"]), box_size(o["bbox"])):
            dx, dy = cx - rx, cy - ry
            horizontal = "오른쪽" if dx > 0 else "왼쪽"
            vertical = "아래" if dy > 0 else "위"
            if min(abs(dx), abs(dy)) > 0.5 * max(abs(dx), abs(dy)):
                side = f"{horizontal} {vertical}"             # 대각선: "노트북 왼쪽 아래"
            elif abs(dx) >= abs(dy):
                side = horizontal
            else:
                side = vertical + "쪽"
            candidates.append((distance, o, side))

    if not candidates:
        return screen_position(t["bbox"], shape)
    _, ref, side = min(candidates, key=lambda c: c[0])
    return f"{name_ko(ref['class'])} {side}"


def update_locations(S, shape):
    """지금 있는 물건마다 위치 설명 갱신 (기준 물건에는 숙소 비품도 포함). 사라진 물건은 마지막 위치를 유지"""
    present = [t for t in S["tracks"] if t["status"] == "present"]
    for t in present:
        t["where"] = describe_location(t, present, shape)
