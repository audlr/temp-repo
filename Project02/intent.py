"""의도 분석(Gemma JSON 출력 + 키워드 규칙), 모르는 물건 이름 붙이기(Gemma 이미지 입력)"""
import re
import json
import base64

from config import *
from common import name_ko


llm = None
if USE_LLM:
    from llama_cpp import Llama
    chat_handler = None
    if USE_VISION_LLM:                       # 이미지 입력용 mmproj (06-B)
        from llama_cpp.llama_chat_format import Gemma4ChatHandler
        chat_handler = Gemma4ChatHandler(clip_model_path=MMPROJ_PATH)
    llm = Llama(
        model_path=GEMMA_MODEL_PATH,
        chat_handler=chat_handler,
        n_gpu_layers=-1,
        n_ctx=CONTEXT_WINDOW,
        n_batch=32,
        n_ubatch=32,
        verbose=False,
    )

ALL_ITEMS = YOLO_ITEMS + list(MARKERS.keys())
GENERIC_WORDS = {"물건", "소지품", "케이스"}
INTENTS = ["check", "locate", "list", "exception_add", "exception_remove", "checkin", "depart",
           "away_on", "away_off", "away_report", "unknown"]


def generate(messages, max_tokens):
    response = llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=0.1)
    return response["choices"][0]["message"]["content"].strip()


def parse_intent_llm(text):
    """사용자 발화 → {"intent": ..., "items": [...]} (Structured Output)"""
    item_list = ", ".join(f"{key}({name_ko(key)})" for key in ALL_ITEMS)
    system_prompt = f"""
Instruction:
숙소 체크아웃 도우미에게 한 사용자 발화를 분석하여 의도와 물건을 JSON으로 출력하시오.

Context:
의도 목록:
- check: 나가도 되는지, 빠뜨린 물건이 없는지 확인 ("나 이제 나가도 돼?", "빠진 거 없어?")
- locate: 특정 물건의 위치를 물음 ("내 폰 어디 있어?")
- list: 내 물건 목록을 물음 ("내 물건 뭐 있어?")
- exception_add: 두고 가도 되는 물건을 알려줌 ("우산은 두고 가도 돼", "컵은 숙소 거야")
- exception_remove: 예외를 취소함 ("우산도 챙길래")
- checkin: 체크인, 지금 있는 물건은 숙소 비품이라고 알려줌 ("체크인", "지금 있는 건 숙소 거야")
- depart: 체크아웃하고 숙소를 떠난다고 알림 ("나 이제 나갈게", "출발할게")
- away_on: 짐을 두고 잠깐 외출함 ("외출할게", "잠깐 나갔다 올게")
- away_off: 외출에서 돌아옴 ("다녀왔어", "나 왔어")
- away_report: 외출한 동안 생긴 일을 물음 ("나 없는 동안 무슨 일 있었어?")
- unknown: 위에 해당하지 않음
물건 키 목록: {item_list}

Constraint:
items에는 물건 키 목록에 있는 키만 넣으시오. 해당 물건이 없으면 빈 리스트로 두시오.
JSON 외의 다른 문장은 출력하지 마시오.

Output Format:
{{"intent": "의도", "items": ["물건 키"]}}
"""
    content = generate(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
        MAX_TOKENS_INTENT,
    )
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        return None
    try:
        result = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if result.get("intent") not in INTENTS:
        return None
    result["items"] = [i for i in result.get("items", []) if i in ALL_ITEMS]
    return result


def parse_intent_rule(text):
    """Gemma를 쓰지 않거나 JSON 파싱에 실패했을 때 쓰는 키워드 규칙"""
    items = [key for key in ALL_ITEMS if name_ko(key) in text or key in text]
    if "폰" in text and "cell phone" not in items:
        items.append("cell phone")

    if "체크인" in text or "숙소 거" in text:
        intent = "checkin" if not items else "exception_add"
    elif any(k in text for k in ["없는 동안", "무슨 일"]):
        intent = "away_report"
    elif any(k in text for k in ["다녀왔", "나 왔어", "돌아왔"]):
        intent = "away_off"
    elif any(k in text for k in ["외출", "나갔다 올", "다녀올게"]):
        intent = "away_on"
    elif any(k in text for k in ["두고 가도", "놓고 가도", "안 챙겨도", "필요 없"]):
        intent = "exception_add"
    elif any(k in text for k in ["챙길", "가져갈"]) and items:
        intent = "exception_remove"
    elif any(k in text for k in ["어디", "위치"]):
        intent = "locate"
    elif any(k in text for k in ["나갈게", "출발할게", "간다"]):
        intent = "depart"
    elif any(k in text for k in ["나가도", "빠진", "빠뜨", "두고 가는", "체크아웃", "다 챙겼"]):
        intent = "check"
    elif any(k in text for k in ["목록", "뭐 있", "뭐뭐"]):
        intent = "list"
    else:
        intent = "unknown"
    return {"intent": intent, "items": items}


def find_named_items(text, state):
    """Gemma가 이름 붙인 모르는 물건 중 발화에 나온 것 → 'id:N' 키 ("충전기 어디 있어?" → 검은색 충전기 케이블)"""
    found = []
    for item in (state or {}).get("items", []):
        if item["class"] != UNKNOWN_CLASS or item["name"].startswith("물건 "):
            continue                                  # Gemma가 이름을 붙이지 않은 물건("물건 3번")은 제외
        words = [w for w in item["name"].split()
                 if len(w) >= 2 and not w.endswith("색") and w not in GENERIC_WORDS]
        if any(w in text for w in words):
            found.append(f"id:{item['id']}")
    return found


def parse_intent(text, state):
    result = parse_intent_llm(text) if USE_LLM else None
    result = result or parse_intent_rule(text)
    result["items"] += [k for k in find_named_items(text, state) if k not in result["items"]]
    return result


def name_item_image(image_path):
    """모르는 물건 ROI 이미지 → Gemma(mmproj)가 짧은 이름 생성 (06-B 이미지 + 텍스트 입력)"""
    with open(image_path, "rb") as file:
        image_base64 = base64.b64encode(file.read()).decode("utf-8")
    image_data = "data:image/jpeg;base64," + image_base64

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": """
Instruction:
이미지 속 물건 하나의 이름을 짧게 말하시오.

Constraint:
여행자의 소지품 이름으로 답하시오. (예: 검은색 충전기, 흰색 이어폰 케이스, 파란 지갑)
무엇인지 확실하지 않으면 "알 수 없음"이라고만 답하시오.
다른 설명이나 문장을 추가하지 마시오.

Output Format:
색 + 물건 이름, 15자 이내 명사.
""",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "이 물건은 무엇인가요?"},
                    {"type": "image_url", "image_url": {"url": image_data}},
                ],
            },
        ],
        max_tokens=MAX_TOKENS_NAME,
        temperature=0.0,
    )
    answer = response["choices"][0]["message"]["content"].strip()
    return clean_name(answer)


def clean_name(answer):
    """모델 답변에서 이름만 남김: 첫 줄, 따옴표·마침표·'입니다' 제거, 길이 제한"""
    name = answer.splitlines()[0] if answer else ""
    name = re.sub(r"[\"'`.*:]|입니다|이에요|예요", "", name).strip()
    if not name or "알 수 없" in name:
        return None
    return name[:15]
