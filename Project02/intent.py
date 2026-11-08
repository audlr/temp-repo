"""의도 분석: Gemma가 발화를 JSON으로 구조화, 실패하거나 Gemma를 끄면 키워드 규칙 사용"""
import re
import json

from config import *
from common import name_ko


llm = None
if USE_LLM:
    from llama_cpp import Llama
    llm = Llama(
        model_path=GEMMA_MODEL_PATH,
        n_gpu_layers=-1,
        n_ctx=CONTEXT_WINDOW,
        n_batch=32,
        n_ubatch=32,
        verbose=False,
    )

ALL_ITEMS = YOLO_ITEMS + list(MARKERS.keys())
INTENTS = ["checkin", "checkout", "locate", "list", "exception_add", "exception_remove", "unknown"]


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
- checkin: 체크인, 지금 있는 물건은 숙소 비품 ("체크인", "체크인할게")
- checkout: 체크아웃, 나가려고 함, 빠뜨린 물건 확인 ("체크아웃", "나 이제 나가도 돼?", "빠진 거 없어?")
- locate: 특정 물건의 위치를 물음 ("내 폰 어디 있어?")
- list: 내 물건 목록을 물음 ("내 물건 뭐 있어?")
- exception_add: 두고 가도 되는 물건을 알려줌 ("우산은 두고 가도 돼")
- exception_remove: 예외를 취소함 ("우산도 챙길래")
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
    compact = text.replace(" ", "")
    items = [key for key in ALL_ITEMS if name_ko(key) in text or key in text]
    if "폰" in text and "cell phone" not in items:
        items.append("cell phone")

    if "체크인" in compact:
        intent = "checkin"
    elif "체크아웃" in compact:
        intent = "checkout"
    elif any(k in text for k in ["두고 가도", "놓고 가도", "안 챙겨도", "필요 없"]):
        intent = "exception_add"
    elif any(k in text for k in ["챙길", "가져갈"]) and items:
        intent = "exception_remove"
    elif any(k in text for k in ["어디", "위치"]):
        intent = "locate"
    elif any(k in text for k in ["나갈", "나가도", "출발", "빠진", "빠뜨", "다 챙겼"]):
        intent = "checkout"
    elif any(k in text for k in ["목록", "뭐 있", "뭐뭐"]):
        intent = "list"
    else:
        intent = "unknown"
    return {"intent": intent, "items": items}


def parse_intent(text):
    result = parse_intent_llm(text) if USE_LLM else None
    return result or parse_intent_rule(text)
