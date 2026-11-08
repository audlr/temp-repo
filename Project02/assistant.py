"""프로세스 2: 음성(또는 키보드) 질문 → 의도 분석 → 규칙 판단 → 음성 답변, vision 체크아웃 결과 안내"""
import os
import sys
import time
import select

from config import *
from common import write_json, read_json, name_ko, josa, ago_ko, is_left, is_missing_amenity
from intent import parse_intent
from speech import speech_to_text, text_to_speech


def group_by_where(items):
    """[휴대폰(노트북 왼쪽), 여권(컵 아래쪽)] → '노트북 왼쪽에 휴대폰, 컵 아래쪽에 여권'"""
    by_where = {}
    for item in items:
        by_where.setdefault(item["where"], []).append(item["name"])
    return ", ".join(f"{where}에 {', '.join(names)}" for where, names in by_where.items())


def checkout_sentence(left, missing, uncertain=None):
    """체크아웃 안내: 남은 내 물건 + 사라진 비품 + 화면에서 확인되지 않는 내 물건"""
    uncertain = uncertain or []
    sentences = []
    if left:
        sentences.append(f"잠깐만요! {josa(group_by_where(left), ('이', '가'))} 아직 남아 있어요.")
    if missing:
        names = ", ".join(i["name"] for i in missing)
        sentences.append(f"그리고 숙소 비품인 {josa(names, ('이', '가'))} 화면에서 확인되지 않아요. 원래 위치에 있는지 확인해 주세요.")
    if uncertain:
        names = ", ".join(i["name"] for i in uncertain)
        sentences.append(f"화면에서 확인되지 않는 내 물건은 {names}이에요. 실제로 챙기셨는지 확인해 주세요.")
    if not sentences:
        return "빠뜨린 물건이 없어요. 조심히 다녀오세요!"
    return " ".join(sentences)


def decide(intent, state, control, now):
    """의도 + 현재 상태 → 답변 문장. 남은 물건 판단은 LLM이 아니라 코드가 함 (없는 물건을 지어내지 않도록)"""
    name = intent["intent"]
    items = intent["items"]

    if state is None:
        return "카메라 프로그램이 아직 실행되지 않았어요."

    if name == "checkin":
        amenities = [i["name"] for i in state["items"] if i["owner"] == "amenity" and i["status"] == "present"]
        listed = f" 숙소 비품은 {josa(', '.join(amenities), ('으로', '로'))} 기억할게요." if amenities else ""
        return "체크인했어요." + listed + " 이제 짐을 풀어 주세요."

    if state["mode"] != "WATCH":
        return "아직 체크인 전이에요. 먼저 '체크인'이라고 말씀해 주세요."

    if name == "checkout":
        left = [i for i in state["items"] if is_left(i, control["exceptions"])]
        missing = [i for i in state["items"] if is_missing_amenity(i)]
        uncertain = [i for i in state["items"] if i["status"] == "missing" and i["owner"] == "mine"]
        return checkout_sentence(left, missing, uncertain)

    if name == "locate":
        if not items:
            return "어떤 물건을 찾으시는지 다시 말씀해 주세요."
        answers = []
        for key in items:
            found = [i for i in state["items"] if i["class"] == key]
            if not found:
                answers.append(f"{josa(name_ko(key), ('은', '는'))} 아직 본 적이 없어요.")
                continue
            item = max(found, key=lambda i: i["last_seen"])
            if item["status"] == "present":
                answers.append(f"{josa(item['name'], ('은', '는'))} 지금 {item['where']}에 있어요.")
            elif item["status"] == "missing":
                answers.append(f"{josa(item['name'], ('은', '는'))} 마지막으로 {item['where']}에서 확인됐지만 지금은 화면에 보이지 않아요. 챙기셨는지 확인해 주세요.")
            else:
                answers.append(f"{josa(item['name'], ('은', '는'))} {ago_ko(item['taken_at'], now)} "
                               f"{item['where']}에서 챙기신 뒤로 안 보여요.")
        return " ".join(answers)

    if name == "list":
        mine = [i for i in state["items"] if i["owner"] == "mine"]
        if not mine:
            return "아직 등록된 소지품이 없어요."
        present = ", ".join(i["name"] for i in mine if i["status"] == "present")
        taken = ", ".join(i["name"] for i in mine if i["status"] == "taken")
        uncertain = ", ".join(i["name"] for i in mine if i["status"] == "missing")
        if present and taken:
            return f"지금 꺼내 둔 물건은 {josa(present, ('이고', '고'))}, 챙긴 물건은 {josa(taken, ('이에요', '예요'))}."
        if present and uncertain:
            return f"지금 꺼내 둔 물건은 {present}이고, 화면에서 확인되지 않는 물건은 {uncertain}이에요. 챙기셨는지 확인해 주세요."
        if uncertain:
            return f"화면에서 확인되지 않는 물건은 {uncertain}이에요. 실제로 챙기셨는지 확인해 주세요."
        if present:
            return f"지금 꺼내 둔 물건은 {josa(present, ('이에요', '예요'))}."
        return f"꺼내 둔 물건은 없고, 챙긴 물건은 {josa(taken, ('이에요', '예요'))}."

    if name == "exception_add":
        if not items:
            return "어떤 물건을 두고 가실지 다시 말씀해 주세요."
        names = ", ".join(name_ko(k) for k in items)
        return f"알겠어요. {josa(names, ('은', '는'))} 확인할 때 빼 둘게요."

    if name == "exception_remove":
        if not items:
            return "어떤 물건을 다시 챙기실지 말씀해 주세요."
        names = ", ".join(name_ko(k) for k in items)
        return f"알겠어요. {names}도 다시 확인할게요."

    return "잘 못 알아들었어요. '체크아웃'이나 '내 폰 어디 있어?'처럼 말씀해 주세요."


def send_control(control, command=None):
    control["seq"] += 1
    control["command"] = command
    write_json(CONTROL_JSON, control)


def apply_intent(intent, control):
    """예외 목록·명령을 control.json에 반영. 명령이 없는 질문도 'snapshot'을 보내 새 사진을 찍게 함"""
    name = intent["intent"]
    if name == "exception_add":
        control["exceptions"] = sorted(set(control["exceptions"]) | set(intent["items"]))
    elif name == "exception_remove":
        control["exceptions"] = [c for c in control["exceptions"] if c not in intent["items"]]

    command = {"checkin": "checkin", "checkout": "checkout"}.get(name, "snapshot")
    send_control(control, command)
    return control["seq"]


def wait_for_snapshot(seq):
    """vision이 방금 보낸 요청(seq)까지 반영한 사진 결과를 저장할 때까지 대기"""
    deadline = time.time() + SNAPSHOT_WAIT_SEC
    state = read_json(STATE_JSON)
    while time.time() < deadline:
        if state and state.get("control_seq", 0) >= seq:
            return state
        time.sleep(0.1)
        state = read_json(STATE_JSON)
    print("[주의] vision에서 새 사진 결과를 받지 못했습니다. vision이 실행 중인지 확인하세요.")
    return state


def handle_question(question, control):
    print(f"[질문] {question}")
    intent = parse_intent(question)
    print(f"[의도] {intent}")

    seq = apply_intent(intent, control)
    state = wait_for_snapshot(seq)
    text_to_speech(decide(intent, state, control, time.time()))


def handle_checkout_event(event):
    """카메라 창에서 o 키로 체크아웃했을 때 음성 안내 (음성 체크아웃은 handle_question에서 답함)"""
    text_to_speech(checkout_sentence(event["items_left"], event["missing_amenities"],
                                     event.get("items_uncertain", [])))


def read_line_nonblocking(timeout):
    """터미널 입력이 있으면 한 줄 반환, 없으면 None (Linux select)"""
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.readline() if ready else None


def main():
    control = read_json(CONTROL_JSON) or {"seq": 0, "command": None, "exceptions": []}
    last_event_seq = (read_json(EVENT_JSON) or {}).get("seq", 0)     # 시작 전에 난 이벤트는 무시

    print("Checkout Assistant 시작")
    print("Enter: 음성 질문 녹음" if USE_STT else "질문을 입력하고 Enter")
    print("텍스트를 입력하고 Enter: 텍스트로 질문 / q: 종료\n")

    while True:
        # 1. 카메라 창에서 o 키로 체크아웃했는지 확인
        event = read_json(EVENT_JSON)
        if event and event.get("seq", 0) > last_event_seq:
            last_event_seq = event["seq"]
            if event.get("reason") == "key":
                handle_checkout_event(event)

        # 2. 사용자 입력 확인 (0.1초 대기, CPU를 계속 쓰지 않도록)
        line = read_line_nonblocking(0.1)
        if line is None:
            continue
        line = line.strip()
        if line == "q":
            break
        if line == "" and USE_STT:
            print(f"말씀해 주세요... ({RECORD_SECONDS}초 녹음)")
            line = speech_to_text()
            print(f"[STT] {line or '(인식된 말 없음)'}")
        if line:
            handle_question(line, control)


if __name__ == "__main__":
    main()
    print("[assistant] 종료", flush=True)
    os._exit(0)          # Gemma(llama.cpp)·CUDA 정리 중 멈추는 경우가 있어 바로 종료
