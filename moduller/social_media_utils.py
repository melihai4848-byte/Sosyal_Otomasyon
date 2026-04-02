import json
import re
from pathlib import Path
from typing import Optional

from moduller.output_paths import find_existing_output
from moduller.subtitle_output_utils import find_subtitle_file, list_subtitle_files
from moduller.srt_utils import parse_srt_blocks, read_srt_file


def normalize_whitespace(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def build_base_stem(value: str) -> str:
    stem = str(value or "").replace(".srt", "")
    for suffix in [
        "_grammar_fixed",
        "_raw",
        "_standart",
        "_standard",
        "_tr",
        "_en",
        "_de",
    ]:
        stem = stem.replace(suffix, "")
    return stem[:-1] if stem.endswith("_") else stem


def load_related_json(girdi_dosyasi: Path, suffix: str) -> Optional[dict | list]:
    aday_stemler = []
    if girdi_dosyasi:
        aday_stemler.append(girdi_dosyasi.stem)
        aday_stemler.append(build_base_stem(girdi_dosyasi.stem))

    for stem in dict.fromkeys([item for item in aday_stemler if item]):
        path = find_existing_output(
            f"{stem}{suffix}",
            groups=("youtube", "instagram", "research", "tools"),
            include_json_cache=True,
        )
        if not path:
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return None


def prepare_transcript(girdi_dosyasi: Path, max_karakter: int = 35000) -> str:
    icerik = read_srt_file(girdi_dosyasi)
    bloklar = parse_srt_blocks(icerik)
    satirlar = [f"[{b.timing_line}] {b.text_content}" for b in bloklar if b.is_processable]
    return "\n".join(satirlar)[:max_karakter]


def select_primary_srt(logger, purpose_label: str = "Modul") -> Optional[Path]:
    default_srt = find_subtitle_file("subtitle_tr.srt")
    if default_srt:
        if logger:
            logger.info(f"{purpose_label} icin varsayilan ana SRT kullaniliyor: {default_srt.name}")
        return default_srt

    srt_files = list_subtitle_files()
    if not srt_files:
        if logger:
            logger.error("❌ Cikti klasorunde SRT bulunamadi.")
        return None

    if logger:
        logger.warning("subtitle_tr.srt bulunamadi. Manuel SRT secimi istenecek.")

    print("\n📂 Lutfen analiz edilecek SRT dosyasini secin:")
    for index, file_path in enumerate(srt_files, start=1):
        print(f"[{index}] {file_path.name}")

    try:
        return srt_files[int(input("\n👉 Secim: ").strip()) - 1]
    except Exception:
        if logger:
            logger.error("Gecersiz secim.")
        return None


def select_metadata_language(metadata_data: Optional[dict], tercih: str = "Türkçe") -> dict:
    if not isinstance(metadata_data, dict):
        return {}
    if isinstance(metadata_data.get(tercih), dict):
        return metadata_data[tercih]
    for value in metadata_data.values():
        if isinstance(value, dict):
            return value
    return {}


def _clean_list(items) -> list[str]:
    if not isinstance(items, list):
        return []
    return [normalize_whitespace(item) for item in items if normalize_whitespace(item)]


def build_critic_summary(critic_data: Optional[dict]) -> str:
    if not isinstance(critic_data, dict):
        return "Video Elestirmeni verisi yok."

    rewrite_items = []
    for item in critic_data.get("rewrite_opportunities", [])[:3]:
        if isinstance(item, dict):
            temiz_item = normalize_whitespace(item.get("problem", ""))
            if temiz_item:
                rewrite_items.append(temiz_item)

    return (
        f"Ozet: {critic_data.get('summary', '')}\n"
        f"Guclu yonler: {' | '.join(_clean_list(critic_data.get('strongest_points', []))[:3])}\n"
        f"Buyuk sorunlar: {' | '.join(_clean_list(critic_data.get('biggest_issues', []))[:3])}\n"
        f"Yeniden yazim firsatlari: {' | '.join(rewrite_items)}"
    ).strip()


def build_trim_summary(trim_data: Optional[dict]) -> str:
    if not isinstance(trim_data, dict):
        return "Kesim onerisi verisi yok."

    trim_targets = []
    for item in trim_data.get("trim_targets", [])[:4]:
        if isinstance(item, dict):
            temiz_item = normalize_whitespace(item.get("problem", ""))
            if temiz_item:
                trim_targets.append(temiz_item)

    return (
        f"Ozet: {trim_data.get('summary', '')}\n"
        f"Oncelikli kesiler: {' | '.join(_clean_list(trim_data.get('priority_cuts', []))[:4])}\n"
        f"Trim hedefleri: {' | '.join(trim_targets)}"
    ).strip()


def build_broll_summary(broll_data: Optional[list]) -> str:
    if not isinstance(broll_data, list):
        return "B-roll verisi yok."
    satirlar = []
    for item in broll_data[:4]:
        if not isinstance(item, dict):
            continue
        satirlar.append(
            f"{item.get('timestamp', '')}: {normalize_whitespace(item.get('reason', ''))} | {normalize_whitespace(item.get('english_prompt', ''))[:120]}"
        )
    return "\n".join(satirlar).strip() or "B-roll verisi yok."


def build_metadata_summary(metadata_data: Optional[dict]) -> str:
    secilen = select_metadata_language(metadata_data)
    if not secilen:
        return "YouTube metadata verisi yok."

    description = secilen.get("description", {})
    if not isinstance(description, dict):
        description = {}
    chapters = secilen.get("chapters", [])
    chapter_preview = []
    if isinstance(chapters, list):
        for item in chapters[:4]:
            if not isinstance(item, dict):
                continue
            timestamp = normalize_whitespace(item.get("timestamp", ""))
            title = normalize_whitespace(item.get("title", ""))
            if timestamp and title:
                chapter_preview.append(f"{timestamp} {title}")

    return (
        f"Basliklar: {' | '.join(_clean_list(secilen.get('titles', []))[:3])}\n"
        f"Hook: {normalize_whitespace(description.get('hook', ''))}\n"
        f"Kisimlar: {' | '.join(chapter_preview)}\n"
        f"Hashtagler: {' '.join(_clean_list(secilen.get('hashtags', []))[:8])}"
    ).strip()

