from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


BASE_DIR = Path(__file__).resolve().parent.parent
ROLE_TABLE_PATH = BASE_DIR / "llm_rol_tablosu.md"


@dataclass(frozen=True)
class LLMRoleTableEntry:
    module_number: str
    title: str
    main_enabled: bool
    smart_enabled: bool
    recommended_main: tuple[str, str] | None = None
    recommended_smart: tuple[str, str] | None = None
    main_summary: str = ""
    smart_summary: str = ""
    notes: str = ""


DEFAULT_ROLE_TABLE: dict[str, LLMRoleTableEntry] = {
    "101": LLMRoleTableEntry("101", "Altyazi Olusturucu (TR)", False, False, notes="Whisper kullanir, Main/Smart kullanmaz."),
    "102": LLMRoleTableEntry(
        "102",
        "Gramer Duzenleyici (TR)",
        True,
        False,
        recommended_main=("OLLAMA", "qwen3:14b"),
        main_summary="video-ozel glossary duzeltmesini, gramer, imla ve satir temizligini yapar",
    ),
    "103": LLMRoleTableEntry("103", "Altyazi Cevirmeni (EN-DE)", False, False, notes="Main/Smart yerine ayri ceviri modeli kullanir."),
    "201": LLMRoleTableEntry(
        "201",
        "Video Aciklamasi (Description) Olusturucu (TR-EN-DE)",
        False,
        True,
        recommended_smart=("OLLAMA", "deepseek-v3.1:671b-cloud"),
        smart_summary="description, baslik ve metadata paketini uretir",
    ),
    "202": LLMRoleTableEntry(
        "202",
        "Video Elestirmeni",
        True,
        True,
        recommended_main=("OLLAMA", "qwen3:14b"),
        recommended_smart=("DEEPSEEK", "deepseek-reasoner"),
        main_summary="ilk analitik taslagi ve yapisal islemeyi yapar",
        smart_summary="nihai analiz, yorum ve packaging cilarini yapar",
    ),
    "203": LLMRoleTableEntry(
        "203",
        "B-Roll Prompt Uretici (16:9 Yatay)",
        False,
        True,
        recommended_smart=("OLLAMA", "kimi-k2.5:cloud"),
        smart_summary="sahne bazli B-roll fikirlerini ve promptlarini uretir",
    ),
    "204": LLMRoleTableEntry(
        "204",
        "Thumbnail Prompt Uretici (16:9 Yatay)",
        False,
        True,
        recommended_smart=("OLLAMA", "kimi-k2.5:cloud"),
        smart_summary="ana thumbnail konseptlerini ve gorsel promptlari uretir",
    ),
    "205": LLMRoleTableEntry(
        "205",
        "Muzik Prompt Olusturucu",
        True,
        False,
        recommended_main=("OLLAMA", "qwen3:14b"),
        main_summary="muzik planini ve segment mantigini cikarir",
    ),
    "301": LLMRoleTableEntry(
        "301",
        "Carousel Fikir Uretici",
        True,
        True,
        recommended_main=("OLLAMA", "qwen3:14b"),
        recommended_smart=("DEEPSEEK", "deepseek-reasoner"),
        main_summary="ilk carousel aday havuzunu cikarir",
        smart_summary="en iyi adaylari secer ve final carousel paketini kurar",
    ),
    "302": LLMRoleTableEntry(
        "302",
        "Reels Fikir Uretici",
        True,
        True,
        recommended_main=("OLLAMA", "qwen3:14b"),
        recommended_smart=("OLLAMA", "kimi-k2.5:cloud"),
        main_summary="ilk reel aday havuzunu cikarir",
        smart_summary="en iyi reel adaylarini secer ve final packagingi yapar",
    ),
    "303": LLMRoleTableEntry(
        "303",
        "Story Serisi Fikir Uretici",
        True,
        True,
        recommended_main=("OLLAMA", "qwen3:14b"),
        recommended_smart=("OLLAMA", "deepseek-v3.1:671b-cloud"),
        main_summary="ilk story aday havuzunu cikarir",
        smart_summary="en iyi story adaylarini secer ve final story setini kurar",
    ),
    "304": LLMRoleTableEntry("304", "Etkilesim Planlayici", False, False, notes="Main/Smart kullanmaz."),
    "401": LLMRoleTableEntry(
        "401",
        "YouTube Trends Konu Fikirleri",
        False,
        True,
        recommended_smart=("DEEPSEEK", "deepseek-reasoner"),
        smart_summary="trend sinyallerini ayiklar, konu fikirlerini skorlar ve en guclu video adaylarini cikarir",
    ),
    "402": LLMRoleTableEntry(
        "402",
        "YouTube Analytics Analizi",
        True,
        True,
        recommended_main=("OLLAMA", "qwen3:14b"),
        recommended_smart=("DEEPSEEK", "deepseek-reasoner"),
        main_summary="kanal ve video verisinden ilk teshis ve analitik yorumu cikarir",
        smart_summary="nihai kritik, aksiyon plani ve stratejik oncelikleri netlestirir",
    ),
    "501": LLMRoleTableEntry("501", "Reels Olusturucu", False, False, notes="Whisper ve ffmpeg kullanir, Main/Smart kullanmaz."),
    "502": LLMRoleTableEntry("502", "YouTube Draft Upload Engine", False, False),
    "503": LLMRoleTableEntry("503", "Automatic B-Roll Downloader", False, False),
    "504": LLMRoleTableEntry("504", "Premiere Pro XML Entegrasyonu", False, False),
    "505": LLMRoleTableEntry("505", "Cikti Temizleyici", False, False),
}


def _normalize_header(value: str) -> str:
    text = str(value or "").strip().lower()
    translations = str.maketrans("çğıöşü", "cgiosu")
    return re.sub(r"[^a-z0-9]+", "", text.translate(translations))


def _parse_bool(value: str, default: bool) -> bool:
    text = str(value or "").strip().lower()
    if text in {"evet", "yes", "true", "1", "var", "x"}:
        return True
    if text in {"hayir", "hayır", "no", "false", "0", "yok", "-"}:
        return False
    return default


def _parse_provider_model(value: str) -> tuple[str, str] | None:
    text = str(value or "").strip()
    if not text or text == "-":
        return None
    if ":" not in text:
        return None
    provider, model_name = text.split(":", 1)
    provider = provider.strip().upper()
    model_name = model_name.strip()
    if not provider or not model_name:
        return None
    return provider, model_name


def _split_table_row(line: str) -> list[str]:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return cells


def _is_divider_row(cells: list[str]) -> bool:
    if not cells:
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)


def _extract_markdown_table(lines: list[str]) -> tuple[list[str], list[list[str]]] | None:
    for index, line in enumerate(lines):
        if not line.lstrip().startswith("|"):
            continue
        header_cells = _split_table_row(line)
        if index + 1 >= len(lines):
            continue
        divider_cells = _split_table_row(lines[index + 1])
        if not _is_divider_row(divider_cells):
            continue
        rows: list[list[str]] = []
        cursor = index + 2
        while cursor < len(lines) and lines[cursor].lstrip().startswith("|"):
            row_cells = _split_table_row(lines[cursor])
            if len(row_cells) == len(header_cells):
                rows.append(row_cells)
            cursor += 1
        return header_cells, rows
    return None


def _build_entry_from_payload(payload: dict[str, str]) -> LLMRoleTableEntry | None:
    module_number = str(payload.get("module", "")).strip()
    default_entry = DEFAULT_ROLE_TABLE.get(module_number)
    if not default_entry:
        return None
    return LLMRoleTableEntry(
        module_number=module_number,
        title=str(payload.get("title", "")).strip() or default_entry.title,
        main_enabled=_parse_bool(payload.get("mainenabled", ""), default_entry.main_enabled),
        smart_enabled=_parse_bool(payload.get("smartenabled", ""), default_entry.smart_enabled),
        recommended_main=_parse_provider_model(payload.get("recommendedmain", "")) or default_entry.recommended_main,
        recommended_smart=_parse_provider_model(payload.get("recommendedsmart", "")) or default_entry.recommended_smart,
        main_summary=str(payload.get("mainsummary", "")).strip() or default_entry.main_summary,
        smart_summary=str(payload.get("smartsummary", "")).strip() or default_entry.smart_summary,
        notes=str(payload.get("notes", "")).strip() or default_entry.notes,
    )


def _parse_role_table_blocks(content: str) -> dict[str, LLMRoleTableEntry]:
    parsed: dict[str, LLMRoleTableEntry] = {}
    current: dict[str, str] = {}

    def flush_current() -> None:
        nonlocal current
        entry = _build_entry_from_payload(current)
        if entry:
            parsed[entry.module_number] = entry
        current = {}

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                flush_current()
            continue
        if line.startswith("#") or line.startswith("- "):
            continue
        if set(line) <= {"-"} and len(line) >= 3:
            if current:
                flush_current()
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = _normalize_header(key)
        if not normalized_key:
            continue
        cleaned_value = value.strip().rstrip("|").strip()
        if normalized_key == "module" and current:
            flush_current()
        current[normalized_key] = cleaned_value

    if current:
        flush_current()

    return parsed


def _parse_role_table_markdown(content: str) -> dict[str, LLMRoleTableEntry]:
    parsed: dict[str, LLMRoleTableEntry] = {}
    extracted = _extract_markdown_table(content.splitlines())
    if not extracted:
        return parsed

    header_cells, rows = extracted
    header_map = {_normalize_header(name): idx for idx, name in enumerate(header_cells)}
    required_headers = {"module", "title", "mainenabled", "smartenabled", "recommendedmain", "recommendedsmart"}
    if not required_headers.issubset(set(header_map)):
        return parsed

    for row in rows:
        payload = {
            "module": row[header_map["module"]],
            "title": row[header_map["title"]],
            "mainenabled": row[header_map["mainenabled"]],
            "smartenabled": row[header_map["smartenabled"]],
            "recommendedmain": row[header_map["recommendedmain"]],
            "recommendedsmart": row[header_map["recommendedsmart"]],
            "mainsummary": row[header_map["mainsummary"]] if "mainsummary" in header_map else "",
            "smartsummary": row[header_map["smartsummary"]] if "smartsummary" in header_map else "",
            "notes": row[header_map["notes"]] if "notes" in header_map else "",
        }
        entry = _build_entry_from_payload(payload)
        if entry:
            parsed[entry.module_number] = entry
    return parsed


def load_llm_role_table_entries() -> dict[str, LLMRoleTableEntry]:
    entries = dict(DEFAULT_ROLE_TABLE)
    if not ROLE_TABLE_PATH.exists():
        return entries
    try:
        content = ROLE_TABLE_PATH.read_text(encoding="utf-8")
        overrides = _parse_role_table_blocks(content)
        if not overrides:
            overrides = _parse_role_table_markdown(content)
    except Exception:
        return entries
    entries.update(overrides)
    return entries
