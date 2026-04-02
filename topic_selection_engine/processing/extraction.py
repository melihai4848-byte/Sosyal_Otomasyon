from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List

from topic_selection_engine.models import RawSignal


TOPIC_TAXONOMY: Dict[str, List[str]] = {
    "almanya_yasam_maliyeti": ["yasam maliyeti", "market", "fatura", "masraf", "gider", "cost of living", "2500 euro", "maas yeter"],
    "almanya_kira_ve_ev": ["kira", "ev bulmak", "wg", "deposit", "schufa", "rent", "wohnung"],
    "almanya_burokrasi": ["burokrasi", "auslanderbehorde", "oturum", "randevu", "anmeldung", "evrak", "residence permit"],
    "almanya_is_ve_almanca": ["is bulmak", "job", "almanca sart", "english only", "warehouse", "gastronomy", "mavi yaka"],
    "almanya_ogrenci_hayati": ["ogrenci", "student", "dorm", "yurt", "wg", "uni", "kampus"],
}

AUDIENCE_PATTERNS: Dict[str, List[str]] = {
    "beginners": ["beginner", "newbie", "sifirdan", "yeni baslayan", "ilk kez"],
    "professionals": ["career", "professional", "is hayati", "kariyer", "work", "office"],
    "creators": ["creator", "youtube", "filmmaking", "editing", "content"],
    "entrepreneurs": ["startup", "founder", "girisim", "business", "sirket"],
    "buyers": ["review", "comparison", "karsilastirma", "best", "price", "fiyat"],
    "travelers": ["travel", "seyahat", "gez", "trip", "tatil"],
    "new_arrivals": ["newcomer", "ilk 30 gun", "new arrivals", "yeni gelen"],
    "workers": ["is", "job", "maas", "worker", "mavi yaka", "office work"],
    "students": ["student", "ogrenci", "dorm", "uni"],
    "families": ["family", "cocuk", "kindergarten", "aile"],
    "future_migrants": ["move", "tasinmak", "gelmek", "planning to move"],
}

EMOTION_PATTERNS: Dict[str, List[str]] = {
    "fear": ["yetmiyor", "zor", "kriz", "delay", "anxiety", "risk", "pahali", "fear"],
    "curiosity": ["ne kadar", "gercek", "mi", "how", "what if", "neden"],
    "opportunity": ["sans", "firsat", "opportunity", "avantaj"],
    "comparison": ["vs", "compare", "comparison", "mi yoksa", "daha iyi"],
}

PAIN_PATTERNS = ["zor", "bekleme", "delay", "yetmiyor", "pahali", "belirsizlik", "anxiety", "kriz", "bulmak zor"]
QUESTION_WORDS = ["mi", "midir", "nasil", "neden", "what", "how", "is", "can"]
STOPWORDS = {
    "ve", "ile", "icin", "ama", "gibi", "the", "and", "for", "how", "what", "almanya", "germany",
    "turkler", "turkish", "people", "video", "users", "about", "this", "that", "ne", "bir", "da", "de",
}


def _extract_keywords(text: str, limit: int = 8) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9']{3,}", text.lower())
    freq = Counter(token for token in tokens if token not in STOPWORDS)
    return [item for item, _ in freq.most_common(limit)]


def _extract_questions(text: str) -> List[str]:
    parts = re.split(r"[.!?\n]", text)
    questions = []
    for part in parts:
        normalized = part.strip()
        if not normalized:
            continue
        lower = normalized.lower()
        if "?" in normalized or any(word in lower.split() for word in QUESTION_WORDS):
            questions.append(normalized)
    return questions[:5]


def _fallback_topic_from_keywords(keywords: List[str]) -> str:
    selected = [keyword for keyword in keywords if keyword not in {"almanya", "germany", "turkish"}][:2]
    if not selected:
        return "topic_general"
    return "topic_" + "_".join(selected)


def extract_signal_features(signal: RawSignal) -> Dict[str, List[str] | str]:
    text = f"{signal.title} {signal.text}".lower()
    keywords = _extract_keywords(text)
    canonical_topics = []
    aliases = []

    for canonical_topic, aliases_list in TOPIC_TAXONOMY.items():
        if any(alias in text for alias in aliases_list):
            canonical_topics.append(canonical_topic)
            aliases.extend([alias for alias in aliases_list if alias in text])

    if not canonical_topics:
        canonical_topics.append(_fallback_topic_from_keywords(keywords))

    audiences = [
        audience
        for audience, patterns in AUDIENCE_PATTERNS.items()
        if any(pattern in text for pattern in patterns)
    ] or ["general_audience"]

    emotions = [
        emotion
        for emotion, patterns in EMOTION_PATTERNS.items()
        if any(pattern in text for pattern in patterns)
    ] or ["curiosity"]

    pain_points = [pattern for pattern in PAIN_PATTERNS if pattern in text]

    return {
        "canonical_topics": canonical_topics,
        "aliases": list(dict.fromkeys(aliases))[:6],
        "audiences": audiences,
        "emotion_triggers": emotions,
        "pain_points": pain_points[:5],
        "questions": _extract_questions(signal.text),
        "keywords": keywords,
    }
