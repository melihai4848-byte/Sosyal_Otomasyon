import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from moduller.srt_utils import SrtBlock, serialize_srt_blocks


_CJK_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿĞğİıŞşÇçÖöÜüÄäßẞ]+")
_META_LEAK_MARKERS = (
    "could you clarify",
    "could you rephrase",
    "i'm here to help",
    "i am here to help",
    "let me know",
    "the phrase",
    "if the original sentence",
    "grammatically correct",
    "please provide the full sentence",
    "your message got cut off",
    "mixed with some unrelated text",
    "specific issue you're facing",
    "however, if there's",
    "it seems like your message",
    "könnten sie",
    "ich bin hier, um zu helfen",
    "bitte geben sie den vollständigen satz",
    "bitte geben sie den vollstandigen satz",
)
_TURKISH_STOPWORDS = {
    "ve",
    "bir",
    "bu",
    "ama",
    "cunku",
    "çünkü",
    "icin",
    "için",
    "ile",
    "cok",
    "çok",
    "daha",
    "gibi",
    "olan",
    "olarak",
    "simdi",
    "şimdi",
    "zaten",
    "degil",
    "değil",
    "sen",
    "siz",
    "biz",
    "ben",
    "hemen",
    "hic",
    "hiç",
    "hicbir",
    "hiçbir",
    "bugun",
    "bugün",
    "sonra",
    "once",
    "önce",
}
_ENGLISH_STOPWORDS = {
    "the",
    "and",
    "you",
    "your",
    "with",
    "for",
    "that",
    "this",
    "are",
    "not",
    "have",
    "will",
    "from",
    "about",
    "please",
    "could",
    "would",
    "help",
    "message",
    "original",
    "sentence",
}
_GERMAN_STOPWORDS = {
    "und",
    "der",
    "die",
    "das",
    "mit",
    "nicht",
    "dass",
    "sie",
    "ich",
    "wir",
    "aber",
    "bitte",
    "konnten",
    "könnten",
    "frage",
    "nachricht",
    "satz",
    "hilfe",
    "ist",
    "eine",
    "einen",
}


def normalize_whitespace(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_text_lines(lines: Any) -> list[str]:
    if isinstance(lines, str):
        source_lines = str(lines).splitlines()
    elif isinstance(lines, list):
        source_lines = [str(item) for item in lines]
    else:
        source_lines = []

    cleaned = [normalize_whitespace(line) for line in source_lines]
    return [line for line in cleaned if line]


def build_subtitle_block_payload(blocks: list[SrtBlock]) -> dict[str, Any]:
    payload_blocks = []
    for block in blocks:
        payload_blocks.append(
            {
                "id": block.id,
                "timing": block.timing_line or "",
                "line_count": len(block.text_lines or []),
                "text_lines": list(block.text_lines or []),
                "text": block.text_content,
            }
        )
    return {"blocks": payload_blocks}


def dump_subtitle_block_payload(blocks: list[SrtBlock]) -> str:
    return json.dumps(build_subtitle_block_payload(blocks), ensure_ascii=False, indent=2)


def _strip_code_fences(text: str) -> str:
    return re.sub(r"```(?:json)?\s*|```", "", str(text or ""), flags=re.IGNORECASE).strip()


def _extract_first_json_value(text: str):
    cleaned = _strip_code_fences(text)
    if not cleaned:
        return None

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    start_positions = [index for index, char in enumerate(cleaned) if char in "{["]
    for start in start_positions:
        stack: list[str] = []
        in_string = False
        escape = False

        for idx in range(start, len(cleaned)):
            char = cleaned[idx]

            if escape:
                escape = False
                continue

            if char == "\\":
                escape = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if char in "{[":
                stack.append(char)
                continue

            if char in "}]":
                if not stack:
                    break
                opener = stack[-1]
                if (opener == "{" and char != "}") or (opener == "[" and char != "]"):
                    break
                stack.pop()
                if not stack:
                    try:
                        return json.loads(cleaned[start : idx + 1])
                    except Exception:
                        break

    return None


def extract_structured_blocks(raw_text: str) -> list[dict[str, Any]] | None:
    payload = _extract_first_json_value(raw_text)
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return None

    for key in ("blocks", "subtitles", "entries", "items", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return None


def _stopword_hits(text: str, stopwords: set[str]) -> int:
    words = [word.casefold() for word in _WORD_RE.findall(str(text or ""))]
    return sum(1 for word in words if word in stopwords)


def _text_quality_issues(
    original_text: str,
    candidate_text: str,
    target_language_code: str,
) -> list[str]:
    issues = []
    normalized = normalize_whitespace(candidate_text)
    lowered = normalized.casefold()

    if not normalized:
        issues.append("empty")
        return issues

    if _CJK_RE.search(normalized):
        issues.append("cjk_chars")

    if any(marker in lowered for marker in _META_LEAK_MARKERS):
        issues.append("meta_leak")

    if "```" in normalized or "<think>" in lowered or "</think>" in lowered:
        issues.append("meta_markup")

    original_length = len(normalize_whitespace(original_text))
    allowed_growth = max(120, original_length * 3)
    if original_length and len(normalized) > allowed_growth:
        issues.append("length_blowup")

    if target_language_code == "tr":
        if _stopword_hits(normalized, _ENGLISH_STOPWORDS) >= 4:
            issues.append("english_leak")
        if _stopword_hits(normalized, _GERMAN_STOPWORDS) >= 4:
            issues.append("german_leak")
    elif target_language_code == "en":
        if _stopword_hits(normalized, _TURKISH_STOPWORDS) >= 3:
            issues.append("turkish_leak")
        if _stopword_hits(normalized, _GERMAN_STOPWORDS) >= 4:
            issues.append("german_leak")
    elif target_language_code == "de":
        if _stopword_hits(normalized, _TURKISH_STOPWORDS) >= 3:
            issues.append("turkish_leak")
        if _stopword_hits(normalized, _ENGLISH_STOPWORDS) >= 4:
            issues.append("english_leak")

    return list(dict.fromkeys(issues))


def _chunk_level_issues(
    original_blocks: list[SrtBlock],
    replacement_lines: list[list[str]],
    target_language_code: str,
) -> list[str]:
    if target_language_code not in {"en", "de"}:
        return []

    comparable = 0
    unchanged = 0
    for original, lines in zip(original_blocks, replacement_lines):
        original_text = normalize_whitespace(original.text_content)
        candidate_text = normalize_whitespace(" ".join(lines))
        if len(original_text) < 18:
            continue
        comparable += 1
        if original_text.casefold() == candidate_text.casefold():
            unchanged += 1

    if comparable >= 4 and unchanged / comparable > 0.35:
        return [f"unchanged_source_ratio:{unchanged}/{comparable}"]
    return []


def validate_structured_subtitle_response(
    raw_text: str,
    expected_blocks: list[SrtBlock],
    target_language_code: str,
) -> list[list[str]]:
    parsed_blocks = extract_structured_blocks(raw_text)
    if not isinstance(parsed_blocks, list):
        raise ValueError("structured_json_missing")

    if len(parsed_blocks) != len(expected_blocks):
        raise ValueError(f"block_count_mismatch:{len(parsed_blocks)}!={len(expected_blocks)}")

    replacement_lines: list[list[str]] = []
    issues: list[str] = []

    for position, (expected_block, item) in enumerate(zip(expected_blocks, parsed_blocks), start=1):
        if not isinstance(item, dict):
            issues.append(f"{expected_block.id or position}:invalid_block_type")
            replacement_lines.append([])
            continue

        item_id = normalize_whitespace(
            item.get("id", item.get("block_id", item.get("index", item.get("subtitle_id", ""))))
        )
        if item_id != expected_block.id:
            issues.append(f"{expected_block.id or position}:id_mismatch:{item_id or 'empty'}")

        lines = None
        for key in ("text_lines", "lines", "translated_lines", "corrected_lines"):
            if key in item:
                lines = item.get(key)
                break

        if lines is None:
            for key in ("text", "translated_text", "corrected_text", "content"):
                if key in item:
                    lines = item.get(key)
                    break

        cleaned_lines = normalize_text_lines(lines)
        replacement_lines.append(cleaned_lines)

        candidate_text = "\n".join(cleaned_lines).strip()
        block_issues = _text_quality_issues(expected_block.text_content, candidate_text, target_language_code)
        if block_issues:
            issues.append(f"{expected_block.id or position}:{'/'.join(block_issues)}")

    issues.extend(_chunk_level_issues(expected_blocks, replacement_lines, target_language_code))
    if issues:
        raise ValueError("; ".join(issues))

    return replacement_lines


def rebuild_srt_from_replacements(
    original_blocks: list[SrtBlock],
    replacement_lines: list[list[str]],
) -> str:
    rebuilt_blocks = []
    for original, new_lines in zip(original_blocks, replacement_lines):
        cleaned_lines = normalize_text_lines(new_lines) or normalize_text_lines(original.text_lines)
        raw = "\n".join([original.index_line or "", original.timing_line or "", *cleaned_lines]).strip()
        rebuilt_blocks.append(
            SrtBlock(
                raw=raw,
                index_line=original.index_line,
                timing_line=original.timing_line,
                text_lines=cleaned_lines,
            )
        )
    return serialize_srt_blocks(rebuilt_blocks).strip()


def prepare_debug_file(path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"=== {title} ===\nBaslangic: {datetime.now().isoformat(timespec='seconds')}\n\n",
        encoding="utf-8",
    )


def append_debug_response(
    path: Path,
    heading: str,
    issues: str,
    raw_response: str,
    source_excerpt: str = "",
) -> None:
    with open(path, "a", encoding="utf-8") as file_obj:
        file_obj.write(f"[{datetime.now().isoformat(timespec='seconds')}] {heading}\n")
        file_obj.write(f"Sorun: {issues}\n")
        if source_excerpt:
            file_obj.write("Kaynak Onizleme:\n")
            file_obj.write(source_excerpt.strip() + "\n")
        file_obj.write("Ham LLM Cevabi:\n")
        file_obj.write(str(raw_response or "").strip() + "\n")
        file_obj.write("\n" + ("-" * 80) + "\n\n")
