# moduller/broll_onerici.py
import json
import re
from pathlib import Path
from typing import Optional

from moduller._module_alias import load_numbered_module
from moduller.logger import get_logger
from moduller.output_paths import stem_json_output_path, txt_output_path
from moduller.social_media_utils import (
    build_critic_summary,
    build_trim_summary,
    load_related_json,
    select_primary_srt,
)
from moduller.srt_utils import read_srt_file, parse_srt_blocks
from moduller.llm_manager import (
    CentralLLM,
    get_module_recommended_llm_config,
    print_module_llm_choice_summary,
    prompt_module_llm_plan,
    select_llm,
)
from moduller.youtube_llm_profiles import call_with_youtube_profile

_ANALYTICS_FEEDBACK_MODULE = load_numbered_module("402_analitik_geri_bildirim_dongusu.py")
load_latest_feedback_data = _ANALYTICS_FEEDBACK_MODULE.load_latest_feedback_data
build_feedback_summary = _ANALYTICS_FEEDBACK_MODULE.build_feedback_summary

logger = get_logger("broll")

SEARCH_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "background",
    "camera",
    "cinematic",
    "close",
    "detailed",
    "dynamic",
    "for",
    "format",
    "footage",
    "from",
    "horizontal",
    "in",
    "into",
    "of",
    "on",
    "over",
    "realistic",
    "scene",
    "shot",
    "slow",
    "the",
    "through",
    "ultra",
    "vertical",
    "video",
    "view",
    "with",
}

TIMING_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})")
BROLL_TRANSCRIPT_MAX_CHARS = 18000
BROLL_MIN_VALID_ITEMS = 3
BROLL_WINDOW_RADIUS = 1
BROLL_TARGET_ANCHORS = 8


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def build_stock_search_query(english_prompt: str, fallback: str = "cinematic stock footage") -> str:
    if not english_prompt:
        return fallback

    cleaned = re.sub(r"[\r\n]+", " ", english_prompt)
    cleaned = re.sub(r"\b(4k|8k|16:9|9:16|hd|uhd|animated|realistic|hyperrealistic)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.split(r"[.;:()\[\]\n]", cleaned, maxsplit=1)[0]
    tokens = re.findall(r"[A-Za-z][A-Za-z'/-]*", cleaned)

    selected: list[str] = []
    for token in tokens:
        normalized = token.lower()
        if normalized in SEARCH_QUERY_STOPWORDS:
            continue
        if len(normalized) <= 2:
            continue
        selected.append(token)
        if len(selected) >= 6:
            break

    query = " ".join(selected).strip()
    return query if query else fallback


def _timing_start_ms(timing_line: str) -> Optional[int]:
    if not timing_line:
        return None
    match = TIMING_RE.search(str(timing_line))
    if not match:
        return None
    hour, minute, second, milli = [int(x) for x in match.groups()]
    return (((hour * 60) + minute) * 60 + second) * 1000 + milli


def _extract_reference_moments(critic_data: Optional[dict], trim_data: Optional[dict]) -> list[int]:
    moments: list[int] = []

    if isinstance(critic_data, dict):
        for key in ("timeline_notes", "rewrite_opportunities"):
            for item in critic_data.get(key, [])[:4]:
                if not isinstance(item, dict):
                    continue
                start_ms = _timing_start_ms(item.get("timestamp", ""))
                if start_ms is not None:
                    moments.append(start_ms)

    if isinstance(trim_data, dict):
        for item in trim_data.get("trim_targets", [])[:5]:
            if not isinstance(item, dict):
                continue
            start_ms = _timing_start_ms(item.get("timestamp", ""))
            if start_ms is not None:
                moments.append(start_ms)

    return moments


def _find_closest_block_index(start_times: list[Optional[int]], target_ms: int) -> Optional[int]:
    best_idx = None
    best_distance = None
    for idx, start_ms in enumerate(start_times):
        if start_ms is None:
            continue
        distance = abs(start_ms - target_ms)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_idx = idx
    return best_idx


def _build_adaptive_transcript(
    girdi_dosyasi: Path,
    max_chars: int = BROLL_TRANSCRIPT_MAX_CHARS,
    critic_data: Optional[dict] = None,
    trim_data: Optional[dict] = None,
) -> str:
    blocks = [b for b in parse_srt_blocks(read_srt_file(girdi_dosyasi)) if b.is_processable]
    if not blocks:
        return ""
    lines = [f"[{b.timing_line}] {b.text_content}" for b in blocks]
    full_text = "\n".join(lines)
    if len(full_text) <= max_chars:
        return full_text

    anchor_indices = {0, min(1, len(blocks) - 1), max(0, len(blocks) - 2), len(blocks) - 1}
    if len(blocks) > 1:
        for idx in range(BROLL_TARGET_ANCHORS):
            fraction = idx / max(BROLL_TARGET_ANCHORS - 1, 1)
            anchor_indices.add(round((len(blocks) - 1) * fraction))

    start_times = [_timing_start_ms(b.timing_line or "") for b in blocks]
    for moment in _extract_reference_moments(critic_data, trim_data):
        closest = _find_closest_block_index(start_times, moment)
        if closest is not None:
            anchor_indices.add(closest)

    ordered_indices = []
    seen = set()
    for idx in sorted(anchor_indices):
        for expanded in range(max(0, idx - BROLL_WINDOW_RADIUS), min(len(blocks), idx + BROLL_WINDOW_RADIUS + 1)):
            if expanded in seen:
                continue
            seen.add(expanded)
            ordered_indices.append(expanded)

    selected_lines: list[str] = []
    current_length = 0
    for idx in ordered_indices:
        line = lines[idx]
        candidate_length = current_length + len(line) + (1 if selected_lines else 0)
        if candidate_length > max_chars and current_length >= int(max_chars * 0.7):
            break
        selected_lines.append(line)
        current_length = candidate_length

    if len(selected_lines) < 8:
        return full_text[:max_chars]

    return "\n".join(selected_lines)


def _routing_decision(critic_data: Optional[dict]) -> tuple[Optional[bool], str]:
    if not isinstance(critic_data, dict):
        return None, ""
    routing = critic_data.get("routing_decisions", {})
    if not isinstance(routing, dict):
        return None, ""
    item = routing.get("broll_generator", {})
    if not isinstance(item, dict):
        return None, ""
    if "run" not in item:
        return None, normalize_whitespace(item.get("reason", ""))
    return bool(item.get("run")), normalize_whitespace(item.get("reason", ""))


def normalize_broll_items(raw_items: list) -> list[dict]:
    normalized: list[dict] = []
    seen_signatures = set()

    for item in raw_items:
        if not isinstance(item, dict):
            continue

        timestamp = normalize_whitespace(item.get("timestamp", ""))
        reason = normalize_whitespace(item.get("reason", ""))
        english_prompt = normalize_whitespace(item.get("english_prompt", ""))

        if not timestamp or not TIMING_RE.search(timestamp):
            continue
        if not reason or len(reason) < 12:
            continue
        if len(english_prompt.split()) < 12:
            continue

        search_query = normalize_whitespace(item.get("stock_search_query", "")) or build_stock_search_query(english_prompt)
        search_query = " ".join(search_query.split()[:6]).strip()
        if not search_query:
            search_query = build_stock_search_query(english_prompt)

        signature = (timestamp, english_prompt[:80].lower())
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        normalized.append(
            {
                "timestamp": timestamp,
                "reason": reason,
                "english_prompt": english_prompt,
                "stock_search_query": search_query,
            }
        )

    return normalized


def clean_and_parse_json(llm_cevabi: str) -> list:
    try:
        cevap = re.sub(r"```json\s*|```", "", str(llm_cevabi or "")).strip()
        start = cevap.find("[")
        end = cevap.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        temiz_json = cevap[start:end + 1]
        return normalize_broll_items(json.loads(temiz_json))
    except Exception as e:
        logger.error(f"JSON Ayrıştırma Hatası: {e}")
        return []


def build_broll_prompt(
    srt_metni: str,
    feedback_ozeti: str,
    critic_ozeti: str = "",
    trim_ozeti: str = "",
    routing_notu: str = "",
) -> str:
    return (
        "Sen uzman bir YouTube kurgucususun.\n"
        "Asagidaki transcript kesitleri videonun farkli bolumlerinden secilmis temsilci parcalardir.\n"
        "Bunlari retention odagi ile analiz et ve sadece gercekten faydali olacak B-Roll anlari oner.\n\n"
        f"AI Critic Ozeti:\n{critic_ozeti or 'Video Elestirmeni verisi yok.'}\n\n"
        f"Trim Ozeti:\n{trim_ozeti or 'Trim verisi yok.'}\n\n"
        f"Retention Geri Bildirim Ozeti:\n{feedback_ozeti}\n\n"
        f"Routing Notu:\n{routing_notu or 'B-Roll adimi normal modda calisiyor.'}\n\n"
        "KRITIK KURALLAR:\n"
        "- SADECE gecerli bir JSON array dondur.\n"
        "- Ortalama her 15-25 saniyede bir veya konu degistiginde B-Roll oner.\n"
        "- Gereksiz yere her satira B-Roll uydurma.\n"
        "- reason alani Turkce olsun ve neden o anda ara goruntu gerektigini aciklasin.\n"
        "- stock_search_query alani yalnizca Ingilizce olsun.\n"
        "- stock_search_query 3 ile 6 kelime arasinda, stok video aramasina uygun, kisa ve net olsun.\n"
        "- english_prompt uzun, sinematik ve detayli kalsin.\n"
        "- english_prompt her sahneyi en az 50 kelime ile betimlesin.\n"
        "- Her promptun sonuna 4K, horizontal, 16:9 format, animated, realistic ekle.\n"
        "- Daha once trim ile zayif oldugu soylenen anlarda B-Roll kullanimi daha oncelikli olabilir.\n\n"
        "JSON FORMAT:\n"
        "[\n"
        "  {\n"
        "    \"timestamp\": \"00:01:20,000 --> 00:01:25,000\",\n"
        "    \"reason\": \"...\",\n"
        "    \"stock_search_query\": \"berlin street view\",\n"
        "    \"english_prompt\": \"...\"\n"
        "  }\n"
        "]\n\n"
        f"Transcript Kesitleri:\n{srt_metni}"
    )


def build_broll_fallback_prompt(
    srt_metni: str,
    feedback_ozeti: str,
    critic_ozeti: str = "",
    trim_ozeti: str = "",
) -> str:
    return (
        "Sadece gecerli bir JSON array dondur.\n"
        "YouTube retention icin B-Roll noktalarini sec.\n"
        "En az 3, en fazla 10 B-Roll oner.\n"
        "Her item timestamp, reason, stock_search_query ve english_prompt alanlarini icermeli.\n"
        "reason Turkce, stock_search_query Ingilizce, english_prompt Ingilizce ve detayli olmali.\n\n"
        f"AI Critic Ozeti:\n{critic_ozeti or 'Yok'}\n\n"
        f"Trim Ozeti:\n{trim_ozeti or 'Yok'}\n\n"
        f"Retention Geri Bildirim Ozeti:\n{feedback_ozeti}\n\n"
        f"Transcript Kesitleri:\n{srt_metni}"
    )


def save_reports(
    girdi_stem: str,
    broll_data: list,
    model_adi: str,
    skip_reason: str = "",
) -> tuple[Path, Path]:
    json_yolu = stem_json_output_path(girdi_stem, "_B_roll_fikirleri.json", group="youtube")
    txt_yolu = txt_output_path("broll")

    with open(json_yolu, "w", encoding="utf-8") as f:
        json.dump(broll_data, f, ensure_ascii=False, indent=2)

    rapor_icerigi = f"=== {girdi_stem} İÇİN B-ROLL KURGU RAPORU ===\n"
    rapor_icerigi += f"Kullanılan Model: {model_adi}\n\n"

    if skip_reason and not broll_data:
        rapor_icerigi += "ROUTING KARARI\n"
        rapor_icerigi += "-" * 50 + "\n"
        rapor_icerigi += f"{skip_reason}\n\n"
        rapor_icerigi += "Bu rapor AI Critic routing kararina gore hafif modda olusturuldu.\n"
        txt_yolu.write_text(rapor_icerigi, encoding="utf-8")
        logger.info(f"🎉 B-Roll raporu kaydedildi: {txt_yolu.name}")
        return json_yolu, txt_yolu

    for i, broll in enumerate(broll_data, 1):
        r_time = broll.get("timestamp", "Bilinmiyor")
        r_reason = broll.get("reason", "Açıklama yok.")
        r_query = broll.get("stock_search_query", "")
        r_prompt = broll.get("english_prompt", "Prompt üretilemedi.")

        rapor_icerigi += f"🎞️ --- B-ROLL KESİTİ {i} ---\n"
        rapor_icerigi += f"⏱️ Zaman Damgası: {r_time}\n"
        rapor_icerigi += f"💡 Neden Konulmalı?: {r_reason}\n"
        if r_query:
            rapor_icerigi += f"🔎 Stok Video Arama Sorgusu:\n{r_query}\n"
        rapor_icerigi += f"🎨 Yapay Zeka / Stok Video Arama Promptu (İngilizce):\n{r_prompt}\n"
        rapor_icerigi += "\n" + "-" * 50 + "\n\n"

    txt_yolu.write_text(rapor_icerigi, encoding="utf-8")
    logger.info(f"🎉 B-Roll raporu kaydedildi: {txt_yolu.name}")
    return json_yolu, txt_yolu


def _generate_broll_data(
    llm: CentralLLM,
    transcript: str,
    feedback_ozeti: str,
    critic_ozeti: str,
    trim_ozeti: str,
    routing_notu: str,
) -> list[dict]:
    prompt = build_broll_prompt(transcript, feedback_ozeti, critic_ozeti, trim_ozeti, routing_notu)
    ham_cevap = call_with_youtube_profile(llm, prompt, profile="creative_ideation")
    broll_data = clean_and_parse_json(ham_cevap)
    if len(broll_data) >= BROLL_MIN_VALID_ITEMS:
        return broll_data

    logger.warning("Ilk B-Roll cevabi yetersiz veya bozuk geldi; strict fallback deneniyor.")
    fallback_prompt = build_broll_fallback_prompt(transcript, feedback_ozeti, critic_ozeti, trim_ozeti)
    fallback_cevap = call_with_youtube_profile(llm, fallback_prompt, profile="strict_json")
    return clean_and_parse_json(fallback_cevap)


def analyze(
    girdi_dosyasi: Path,
    llm: CentralLLM,
    feedback_data=None,
    critic_data: Optional[dict] = None,
    trim_data: Optional[dict] = None,
    prepared_transcript: Optional[str] = None,
    respect_routing: bool = False,
) -> dict:
    active_feedback = feedback_data or load_latest_feedback_data()
    feedback_ozeti = build_feedback_summary(active_feedback)
    active_critic = critic_data or load_related_json(girdi_dosyasi, "_video_critic.json")
    active_trim = trim_data or load_related_json(girdi_dosyasi, "_trim_suggestions.json")
    critic_ozeti = build_critic_summary(active_critic)
    trim_ozeti = build_trim_summary(active_trim)

    routing_run, routing_reason = _routing_decision(active_critic)
    if respect_routing and routing_run is False:
        logger.info("AI Critic B-Roll adimini gereksiz gordu; hafif skip raporu olusturuluyor.")
        return {
            "data": [],
            "skip_reason": routing_reason or "AI Critic ek B-Roll zorunlu gormedi.",
            "detail": routing_reason or "AI Critic ek B-Roll zorunlu gormedi.",
        }

    transcript = normalize_whitespace(prepared_transcript) or _build_adaptive_transcript(
        girdi_dosyasi,
        max_chars=BROLL_TRANSCRIPT_MAX_CHARS,
        critic_data=active_critic,
        trim_data=active_trim,
    )
    routing_notu = routing_reason or "AI Critic tarafindan B-Roll adimi uygun goruldu."
    broll_data = _generate_broll_data(llm, transcript, feedback_ozeti, critic_ozeti, trim_ozeti, routing_notu)

    if not broll_data:
        logger.error("❌ YZ cevabı işlenemedi veya JSON formatı hatalı.")
        return {}

    return {
        "data": broll_data,
        "skip_reason": "",
        "detail": "B-Roll fikirleri uretildi.",
    }


def run():
    print("\n" + "=" * 50)
    print("🎞️ AŞAMA 6: YAPAY ZEKA B-ROLL FİKİR ÜRETİCİ")
    print("=" * 50)

    secilen_srt = select_primary_srt(logger, "B-Roll Prompt Uretici")
    if not secilen_srt:
        return

    use_recommended = prompt_module_llm_plan("203", needs_smart=True)
    if use_recommended:
        saglayici, model_adi = get_module_recommended_llm_config("203", "smart")
        print_module_llm_choice_summary("203", {"smart": (saglayici, model_adi)})
    else:
        saglayici, model_adi = select_llm("smart")
    llm = CentralLLM(provider=saglayici, model_name=model_adi)

    result = analyze(
        secilen_srt,
        llm,
        feedback_data=load_latest_feedback_data(),
        critic_data=load_related_json(secilen_srt, "_video_critic.json"),
        trim_data=load_related_json(secilen_srt, "_trim_suggestions.json"),
        respect_routing=True,
    )
    if not result:
        return logger.error("YZ cevabı işlenemedi veya JSON formatı hatalı.")

    save_reports(secilen_srt.stem, result["data"], model_adi, skip_reason=result.get("skip_reason", ""))


def run_automatic(
    girdi_dosyasi: Path,
    llm: CentralLLM,
    feedback_data=None,
    critic_data: Optional[dict] = None,
    trim_data: Optional[dict] = None,
    prepared_transcript: Optional[str] = None,
    respect_routing: bool = True,
):
    logger.info(f"🔄 OTOMASYON: {girdi_dosyasi.name} için B-Roll fikirleri üretiliyor...")

    result = analyze(
        girdi_dosyasi,
        llm,
        feedback_data=feedback_data,
        critic_data=critic_data,
        trim_data=trim_data,
        prepared_transcript=prepared_transcript,
        respect_routing=respect_routing,
    )
    if not result:
        return None

    json_yolu, txt_yolu = save_reports(
        girdi_dosyasi.stem,
        result["data"],
        llm.model_name,
        skip_reason=result.get("skip_reason", ""),
    )
    return {
        "data": result["data"],
        "json_path": json_yolu,
        "txt_path": txt_yolu,
        "detail": result.get("detail", ""),
        "skipped_by_routing": bool(result.get("skip_reason")),
    }
