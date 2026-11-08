"""추적: 사진 사이 같은 물체 연결, 있음/챙김 판정

track 상태: candidate(확인 중) → present(있음) → taken(챙김) → (다시 보이면) present
owner: unknown(체크인 전) / amenity(숙소 비품) / mine(내 소지품) / ignored(Gemma가 물건이 아니라고 판단)
"""
from collections import deque

from config import *
from common import calculate_iou, track_name, zone_of


def new_track(S, det, now):
    S["next_id"] += 1
    return {"id": S["next_id"], "class": det["class"], "bbox": det["bbox"],
            "hits": deque([1], maxlen=STABLE_WINDOW), "status": "candidate",
            "owner": "mine" if S["mode"] == "WATCH" else "unknown",
            "zone": zone_of(det["bbox"]), "first_seen": now, "last_seen": now,
            "taken_at": None, "origin": None, "label": None, "label_requested": False}


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
    """candidate가 안정적으로 보이면 present로 확정. 같은 클래스의 '챙김' 기록이 있으면 그 기록을 되살림
    (물건을 옮겨 놓은 경우 → IoU 매칭이 끊겨도 같은 물건으로 취급)"""
    taken = [o for o in S["tracks"] if o["class"] == t["class"] and o["status"] == "taken" and o is not t]
    if taken and t["class"] != UNKNOWN_CLASS:          # 모르는 물건끼리는 같은 물건인지 알 수 없으므로 제외
        old = max(taken, key=lambda o: o["taken_at"])
        old.update(bbox=t["bbox"], hits=t["hits"], status="present", zone=t["zone"],
                   last_seen=t["last_seen"], taken_at=None)
        S["tracks"].remove(t)
        return old
    t["status"] = "present"
    return t


def update_tracks(S, detections, now):
    """탐지 결과로 track 갱신. 가방이 사라지면(출발) True 반환"""
    matched, unmatched = match_detections(S["tracks"], detections)
    seen = set()

    for t, d in matched:
        t.update(bbox=d["bbox"], zone=zone_of(d["bbox"]), last_seen=now)
        t["hits"].append(1)
        seen.add(t["id"])

    departed = False
    for t in list(S["tracks"]):
        if t["id"] not in seen:
            t["hits"].append(0)

        if t["status"] == "candidate":
            if sum(t["hits"]) >= STABLE_HITS:
                S["changes"].append(("appeared", confirm(S, t)))
            elif now - t["last_seen"] > MISSING_SEC:
                S["tracks"].remove(t)                           # 잠깐 보였던 오탐지 제거

        elif t["status"] == "present" and now - t["last_seen"] > MISSING_SEC:
            t["status"], t["taken_at"] = "taken", t["last_seen"]
            S["changes"].append(("taken", t))
            if t["class"] in CARRIERS:
                departed = True

    for d in unmatched:
        S["tracks"].append(new_track(S, d, now))

    return departed


def summarize(t):
    return {"id": t["id"], "class": t["class"], "name": track_name(t),
            "zone": t["zone"]["name"], "status": t["status"], "owner": t["owner"],
            "first_seen": t["first_seen"], "last_seen": t["last_seen"], "taken_at": t["taken_at"]}
