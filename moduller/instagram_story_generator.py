import json
import re
import time
from pathlib import Path
from typing import Optional, Tuple

from moduller._module_alias import load_numbered_module
from moduller.llm_manager import (
    CentralLLM,
    get_module_recommended_llm_config,
    print_module_llm_choice_summary,
    prompt_module_llm_plan,
    select_llm,
)
from moduller.logger import get_logger
from moduller.output_paths import grouped_output_path, stem_json_output_path, txt_output_path
from moduller.social_media_utils import (
    build_broll_summary,
    build_critic_summary,
    build_metadata_summary,
    load_related_json,
    prepare_transcript,
    select_primary_srt,
)
from moduller.youtube_llm_profiles import call_with_youtube_profile

_VIDEO_CRITIC_MODULE = load_numbered_module("202_video_critic.py")
extract_json_response = _VIDEO_CRITIC_MODULE.extract_json_response
normalize_whitespace = _VIDEO_CRITIC_MODULE.normalize_whitespace

logger = get_logger("story")
LLM_RETRIES = 3
MIN_STORY_CANDIDATES = 7
MAX_STORY_CANDIDATES = 10
STORY_PROMPT_FOOTER = (
    "Metin buyuk, mobilde rahat okunur olsun, 2-3 satiri gecmesin ve guclu gorsel hiyerarsiyle ana mesaja "
    "odaklansin."
)
STORY_FORMAT_SPEC_TR = "Format/Cozunurluk: 9:16 (dikey) en-boy orani, 1080x1920."
DEFAULT_STORY_GOAL_EN = "Etkilesim ve cevap istegi uyandiran dikkat cekici bir Instagram Story tasarimi."
DEFAULT_STORY_BACKGROUND_EN = "Temiz, modern ve acik tonlu bir arka plan; sosyal medya icin optimize edilmis."
DEFAULT_STORY_SUBJECT_EN = "Konuyla baglantili gercekci bir kisi, nesne veya net bir sahne odagi."
DEFAULT_STORY_SUPPORTING_ELEMENTS_EN = "Ince ikonlar, yonlendirici ipuclari ve ana mesaji guclendiren tek bir vurgu alani."
DEFAULT_STORY_STYLE_EN = "Modern, temiz, yuksek kontrastli ve mobilde kolay okunur bir story dili."
STORY_DEBUG_DIR = grouped_output_path("instagram", "_llm_debug")
STORY_ITEM_TXT_DIR = grouped_output_path("instagram", "303_IG-Story_Fikirleri")
LEGACY_STORY_ITEM_TXT_DIR = STORY_ITEM_TXT_DIR.parent / "_tekil_story_txt"
TURKISH_PROMPT_MARKERS = {
    "acik",
    "amac",
    "ana",
    "arka",
    "arkaplan",
    "baslik",
    "bir",
    "bu",
    "gorsel",
    "icin",
    "metin",
    "neden",
    "ozne",
    "ruh",
    "stil",
    "story",
    "tasarim",
    "unsur",
    "dikey",
    "cozunurluk",
    "format",
    "kaydir",
    "anket",
    "soru",
}


def _first_non_empty(*values: object) -> str:
    for value in values:
        text = normalize_whitespace(value)
        if text:
            return text
    return ""


def _looks_turkish(text: object) -> bool:
    content = normalize_whitespace(text)
    if not content:
        return False
    if any(ch in content for ch in "çğıöşüÇĞİÖŞÜİı"):
        return True
    lowered = content.casefold()
    words = set(re.findall(r"[a-zA-Zçğıöşü]+", lowered))
    matches = sum(1 for marker in TURKISH_PROMPT_MARKERS if marker in words)
    return matches >= 2


def _prefer_turkish_prompt_value(value: object, *fallbacks: object) -> str:
    text = normalize_whitespace(value)
    if text and _looks_turkish(text):
        return text
    for fallback in fallbacks:
        fallback_text = normalize_whitespace(fallback)
        if fallback_text:
            return fallback_text
    return text


def _normalize_story_prompt_fields(item: dict) -> dict:
    story_title_tr = normalize_whitespace(item.get("story_title_tr", ""))
    story_content_tr = normalize_whitespace(item.get("story_content_tr", item.get("story_direction_tr", "")))
    visual_direction_tr = normalize_whitespace(item.get("visual_direction_tr", ""))
    interaction_tip_tr = normalize_whitespace(item.get("interaction_tip_tr", ""))

    goal = _prefer_turkish_prompt_value(
        item.get("goal_en", ""),
        item.get("story_goal_en", ""),
        DEFAULT_STORY_GOAL_EN,
    )
    background = _prefer_turkish_prompt_value(
        item.get("background_en", ""),
        visual_direction_tr,
        DEFAULT_STORY_BACKGROUND_EN,
    )
    subject = _prefer_turkish_prompt_value(
        item.get("subject_en", ""),
        visual_direction_tr,
        story_content_tr,
        DEFAULT_STORY_SUBJECT_EN,
    )
    primary_text = _prefer_turkish_prompt_value(
        item.get("primary_text_en", ""),
        item.get("story_title_en", ""),
        story_title_tr,
    )
    supporting_elements = _prefer_turkish_prompt_value(
        item.get("supporting_elements_en", ""),
        interaction_tip_tr,
        story_content_tr,
        DEFAULT_STORY_SUPPORTING_ELEMENTS_EN,
    )
    style = _prefer_turkish_prompt_value(
        item.get("style_mood_en", ""),
        item.get("style_en", ""),
        visual_direction_tr,
        DEFAULT_STORY_STYLE_EN,
    )

    return {
        "goal_en": goal,
        "background_en": background,
        "subject_en": subject,
        "primary_text_en": primary_text,
        "supporting_elements_en": supporting_elements,
        "style_mood_en": style,
    }


def build_story_image_prompt(item: dict) -> str:
    prompt_fields = _normalize_story_prompt_fields(item)
    lines = [
        "Bir Instagram Story tasarimi olustur.",
        f"Amac: {prompt_fields['goal_en']}",
        f"Arka Plan: {prompt_fields['background_en']}",
        f"Ozne: {prompt_fields['subject_en']}",
        f'Ana Metin: "{prompt_fields["primary_text_en"]}"',
        f"Destekleyici Unsurlar: {prompt_fields['supporting_elements_en']}",
        f"Stil/Ruh Hali: {prompt_fields['style_mood_en']}",
        STORY_FORMAT_SPEC_TR,
        STORY_PROMPT_FOOTER,
    ]
    return "\n".join(line for line in lines if normalize_whitespace(line))


def _safe_debug_label(value: object, default: str = "stage") -> str:
    text = normalize_whitespace(value).casefold()
    cleaned = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return cleaned or default


def _debug_response_path(debug_stem: str, stage_label: str, kind: str) -> Path:
    STORY_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    stem = _safe_debug_label(debug_stem, "story")
    stage = _safe_debug_label(stage_label, "stage")
    suffix = _safe_debug_label(kind, "raw")
    return STORY_DEBUG_DIR / f"{stem}_{stage}_{suffix}.txt"


def _save_debug_response(
    debug_stem: str,
    stage_label: str,
    kind: str,
    response_text: object,
    *,
    note: str = "",
) -> Path | None:
    content = str(response_text or "").strip()
    if not content:
        return None

    path = _debug_response_path(debug_stem, stage_label, kind)
    parts = []
    if note:
        parts.extend([note, ""])
    parts.append(content)
    path.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")
    logger.warning(f"Story debug cevabi kaydedildi: {path.name}")
    return path


def _json_repair_schema_hint(stage_label: str) -> str:
    stage = _safe_debug_label(stage_label, "stage")
    if stage == "ideation":
        return """
Expected root shape:
- `why_this_many_stories_tr`: string
- `story_candidates`: array

Each story candidate should remain an object that includes:
- `story_title_tr`
- `angle_tr`
- `why_selected_tr`
- `story_direction_tr`
- `visual_direction_tr`
- `interaction_tip_tr`
- `goal_en`
- `background_en`
- `subject_en`
- `primary_text_en`
- `supporting_elements_en`
- `style_mood_en`
""".strip()

    return """
Expected root shape:
- `why_this_many_stories_tr`: string
- `selected_story_count`: integer
- `story_candidates`: array

Each story candidate should remain an object that includes:
- `rank`
- `engagement_score`
- `story_title_tr`
- `why_selected_tr`
- `story_content_tr`
- `visual_direction_tr`
- `interaction_tip_tr`
- `goal_en`
- `background_en`
- `subject_en`
- `primary_text_en`
- `supporting_elements_en`
- `style_mood_en`
""".strip()


def _build_invalid_json_repair_prompt(raw_response: str, stage_label: str) -> str:
    schema_hint = _json_repair_schema_hint(stage_label)
    return f"""
You are repairing malformed Instagram story JSON produced by another model.

Task:
Convert the raw assistant output below into one valid JSON object.

Rules:
- Return only JSON. No markdown fences. No explanation.
- Preserve the original meaning and structure as much as possible.
- Keep all audience-facing fields in Turkish.
- Keep the legacy `_en` prompt fields in Turkish too; only the key names stay the same.
- Remove brainstorm prose, duplicate wrappers, and trailing commentary outside JSON.
- Fix invalid commas, brackets, quotes, and escaping.
- If a field is obviously cut off, complete it minimally and conservatively so the JSON becomes valid.
- Do not invent extra story candidates unless required to preserve valid structure.

Schema guidance:
{schema_hint}

RAW OUTPUT:
{raw_response}
""".strip()


def _build_strict_json_retry_prompt(original_prompt: str, stage_label: str) -> str:
    return f"""
The previous response for the Instagram story `{stage_label}` stage was not valid JSON.

Return a fresh answer from scratch.

Hard rules:
- Return only one valid JSON object.
- Do not include analysis, brainstorm text, markdown fences, or any extra text before/after JSON.
- Escape all double quotes correctly inside string values.
- Ensure commas, arrays, and braces are fully valid.
- If you are unsure, keep values shorter and simpler rather than verbose.

Follow these original task instructions exactly:

{original_prompt}
""".strip()


def _repair_invalid_json_response(
    llm: CentralLLM,
    raw_response: str,
    *,
    stage_label: str,
    debug_stem: str,
    attempt_no: int,
) -> Optional[dict]:
    if not normalize_whitespace(raw_response):
        return None

    try:
        repaired_response = call_with_youtube_profile(
            llm,
            _build_invalid_json_repair_prompt(raw_response, stage_label),
            profile="analytic_json",
        )
    except Exception as exc:
        logger.warning(f"Story JSON repair istegi basarisiz oldu ({stage_label} / deneme {attempt_no}): {exc}")
        return None

    _save_debug_response(
        debug_stem,
        stage_label,
        f"attempt_{attempt_no}_repair_response",
        repaired_response,
        note="Otomatik JSON repair cevabi",
    )

    repaired = extract_json_response(repaired_response, logger_override=logger, log_errors=False)
    if repaired:
        logger.info(f"Story JSON repair basarili oldu: {stage_label} (deneme {attempt_no})")
    else:
        logger.warning(f"Story JSON repair gecersiz cevap dondu: {stage_label} (deneme {attempt_no})")
    return repaired


def _retry_with_strict_json_profile(
    llm: CentralLLM,
    prompt: str,
    *,
    stage_label: str,
    debug_stem: str,
    attempt_no: int,
) -> Optional[dict]:
    try:
        strict_response = call_with_youtube_profile(
            llm,
            _build_strict_json_retry_prompt(prompt, stage_label),
            profile="strict_json",
        )
    except Exception as exc:
        logger.warning(f"Story strict_json retry basarisiz oldu ({stage_label} / deneme {attempt_no}): {exc}")
        return None

    _save_debug_response(
        debug_stem,
        stage_label,
        f"attempt_{attempt_no}_strict_raw",
        strict_response,
        note="strict_json profili ile yeniden uretilen ham cevap",
    )

    parsed = extract_json_response(strict_response, logger_override=logger, log_errors=False)
    if parsed:
        logger.info(f"Story strict_json retry basarili oldu: {stage_label} (deneme {attempt_no})")
        return parsed

    repaired = _repair_invalid_json_response(
        llm,
        strict_response,
        stage_label=f"{stage_label}_strict",
        debug_stem=debug_stem,
        attempt_no=attempt_no,
    )
    if repaired:
        return repaired

    logger.warning(f"Story strict_json retry da parse edilemedi: {stage_label} (deneme {attempt_no})")
    return None


def _request_llm(
    llm: CentralLLM,
    prompt: str,
    retries: int = LLM_RETRIES,
    profile: str = "creative_ranker",
    debug_stem: str = "story",
    stage_label: str = "response",
) -> Optional[dict]:
    for deneme in range(1, retries + 1):
        try:
            cevap = call_with_youtube_profile(llm, prompt, profile=profile)
            parsed = extract_json_response(cevap, logger_override=logger, log_errors=False)
            if parsed:
                return parsed
            _save_debug_response(
                debug_stem,
                stage_label,
                f"attempt_{deneme}_invalid_raw",
                cevap,
                note="Ilk ham cevap parse edilemedi",
            )
            repaired = _repair_invalid_json_response(
                llm,
                str(cevap or ""),
                stage_label=stage_label,
                debug_stem=debug_stem,
                attempt_no=deneme,
            )
            if repaired:
                return repaired
            strict_retry = _retry_with_strict_json_profile(
                llm,
                prompt,
                stage_label=stage_label,
                debug_stem=debug_stem,
                attempt_no=deneme,
            )
            if strict_retry:
                return strict_retry
            logger.error("LLM cevabından geçerli JSON çıkarılamadı.")
        except Exception as exc:
            logger.warning(f"Story planlayici LLM hatasi ({deneme}/{retries}): {exc}")
        time.sleep(deneme)
    return None


def build_ideation_prompt(transkript: str, metadata_ozeti: str, broll_ozeti: str, critic_ozeti: str) -> str:
    return f"""
Sen Instagram story stratejisti, sosyal medya kreatif direktoru ve izleyiciyi etkilesime sokan bir icerik editorusun.

Gorevin:
Ana videonun transcriptini okuyup bu konu etrafinda paylasilabilecek 7 ile 10 arasinda farkli story adayi dusun.

KRITIK KURALLAR:
- Final JSON'dan once kisa bir <brainstorm> bolumunde hangi story acilarinin neden calisacagini, hangilerinin anket/soru/slider gibi etkilesimlerle daha iyi akacagini dusun.
- <brainstorm> bolumunde yalnizca duz metin kullan; {{ }}, [ ] veya kod blogu kullanma.
- Final cevabinin sonunda tek bir gecerli JSON nesnesi ya da tek bir ```json``` blogu ver.
- Tum metinler Turkce olsun.
- Gorsel uretim prompt alanlari da dahil olmak uzere tum prompt icerikleri Turkce olsun.
- `_en` eki yalnizca eski alan ismidir; `goal_en`, `background_en`, `subject_en`, `primary_text_en`, `supporting_elements_en` ve `style_mood_en` alanlarinin degerlerini Turkce yaz.
- En az {MIN_STORY_CANDIDATES} story adayi cikar.
- En fazla {MAX_STORY_CANDIDATES} story adayi cikar.
- Her aday ayri bir story mantigi kullansin.
- Her aday icin image generation tarafinda kullanilacak yapisal alanlar da uret.
- Prompt mantigi su formatla uyumlu olsun:
  Bir Instagram Story tasarimi olustur.
  Amac: ...
  Arka Plan: ...
  Ozne: ...
  Ana Metin: "..."
  Destekleyici Unsurlar: ...
  Stil/Ruh Hali: ...
- Her promptta format/cozunurluk bilgisini acikca belirt:
  {STORY_FORMAT_SPEC_TR}
- Asagidaki cümleyi prompt mantigina mutlaka yedir:
  {STORY_PROMPT_FOOTER}

JSON SEMASI:
{{
  "why_this_many_stories_tr": "",
  "story_candidates": [
    {{
      "story_title_tr": "",
      "angle_tr": "",
      "why_selected_tr": "",
      "story_direction_tr": "",
      "visual_direction_tr": "",
      "interaction_tip_tr": "",
      "goal_en": "",
      "background_en": "",
      "subject_en": "",
      "primary_text_en": "",
      "supporting_elements_en": "",
      "style_mood_en": ""
    }}
  ]
}}

YouTube Metadata Ozeti:
{metadata_ozeti}

B-Roll Ozeti:
{broll_ozeti}

Video Elestirmeni Ozeti:
{critic_ozeti}

Transcript:
{transkript}
""".strip()


def build_selection_prompt(
    transkript: str,
    metadata_ozeti: str,
    broll_ozeti: str,
    critic_ozeti: str,
    ideation_payload: dict,
) -> str:
    ideation_text = json.dumps(ideation_payload, ensure_ascii=False, indent=2)
    return f"""
Sen Instagram story stratejisti, sosyal medya kreatif direktoru ve final packaging editorusun.

Gorevin:
Verilen story adaylarini incele, en guclu olanlari sec, etkilesim potansiyellerine gore skorla ve 7 ile 10 arasinda final story fikir listesi olustur.

KRITIK KURALLAR:
- Once aday akislardaki guclu, zayif ve fazla benzer fikirleri kisaca analiz edebilirsin.
- Analiz kismini yalnizca duz metin veya <brainstorm> etiketiyle ver; {{ }}, [ ] veya kod blogu kullanma.
- Analizden sonra cevabini tek bir gecerli JSON nesnesi ya da tek bir ```json``` blogu olarak bitir.
- Tum metinler Turkce olsun.
- `goal_en`, `background_en`, `subject_en`, `primary_text_en`, `supporting_elements_en`, `style_mood_en` alanlari da Turkce olsun.
- `_en` eki yalnizca eski alan ismidir; bu alanlarin degerlerini Turkce yaz.
- En az {MIN_STORY_CANDIDATES} story adayi cikar.
- En fazla {MAX_STORY_CANDIDATES} story adayi cikar.
- Her aday icin neden secildigini, storyde ne olmali bilgisini ve gorselin nasil olmasi gerektigini detayli anlat.
- Image generation prompt yapisi su mantiga tam uysun:
  Bir Instagram Story tasarimi olustur.
  Amac: ...
  Arka Plan: ...
  Ozne: ...
  Ana Metin: "..."
  Destekleyici Unsurlar: ...
  Stil/Ruh Hali: ...
- Her promptta format/cozunurluk bilgisini acikca belirt:
  {STORY_FORMAT_SPEC_TR}
- Asagidaki cümleyi prompt tasariminin bir parcasi olarak mutlaka destekle:
  {STORY_PROMPT_FOOTER}

JSON SEMASI:
{{
  "why_this_many_stories_tr": "",
  "selected_story_count": 0,
  "story_candidates": [
    {{
      "rank": 1,
      "engagement_score": 92,
      "story_title_tr": "",
      "why_selected_tr": "",
      "story_content_tr": "",
      "visual_direction_tr": "",
      "interaction_tip_tr": "",
      "goal_en": "",
      "background_en": "",
      "subject_en": "",
      "primary_text_en": "",
      "supporting_elements_en": "",
      "style_mood_en": ""
    }}
  ]
}}

YouTube Metadata Ozeti:
{metadata_ozeti}

B-Roll Ozeti:
{broll_ozeti}

Video Elestirmeni Ozeti:
{critic_ozeti}

Story Aday Havuzu:
{ideation_text}

Transcript:
{transkript}
""".strip()


def build_prompt(transkript: str, metadata_ozeti: str, broll_ozeti: str, critic_ozeti: str) -> str:
    return f"""
Sen Instagram story stratejisti, sosyal medya kreatif direktoru ve story packaging editorusun.

Gorevin:
Transcripti okuyup bu konu etrafinda paylasilabilecek 7 ile 10 arasinda farkli story adayi cikar, bunlari etkilesim potansiyeline gore skorla ve final detaylarini uret.

KRITIK KURALLAR:
- Final JSON'dan once kisa bir <brainstorm> bolumunde hangi story acilarinin neden daha iyi calisacagini dusun.
- <brainstorm> bolumunde yalnizca duz metin kullan; {{ }}, [ ] veya kod blogu kullanma.
- Analizden sonra cevabini tek bir gecerli JSON nesnesi ya da tek bir ```json``` blogu olarak bitir.
- Tum metinler Turkce olsun.
- `goal_en`, `background_en`, `subject_en`, `primary_text_en`, `supporting_elements_en`, `style_mood_en` alanlari da Turkce olsun.
- `_en` eki yalnizca eski alan ismidir; bu alanlarin degerlerini Turkce yaz.
- En az {MIN_STORY_CANDIDATES} story adayi cikar.
- En fazla {MAX_STORY_CANDIDATES} story adayi cikar.
- Her aday icin neden secildigini, storyde ne olmali bilgisini ve gorselin nasil olmasi gerektigini detayli anlat.
- Image generation prompt mantigi su formatla uyumlu olsun:
  Bir Instagram Story tasarimi olustur.
  Amac: ...
  Arka Plan: ...
  Ozne: ...
  Ana Metin: "..."
  Destekleyici Unsurlar: ...
  Stil/Ruh Hali: ...
- Her promptta format/cozunurluk bilgisini acikca belirt:
  {STORY_FORMAT_SPEC_TR}
- Asagidaki cümleyi prompt mantigina mutlaka ekle:
  {STORY_PROMPT_FOOTER}

JSON SEMASI:
{{
  "why_this_many_stories_tr": "",
  "selected_story_count": 0,
  "story_candidates": [
    {{
      "rank": 1,
      "engagement_score": 92,
      "story_title_tr": "",
      "why_selected_tr": "",
      "story_content_tr": "",
      "visual_direction_tr": "",
      "interaction_tip_tr": "",
      "goal_en": "",
      "background_en": "",
      "subject_en": "",
      "primary_text_en": "",
      "supporting_elements_en": "",
      "style_mood_en": ""
    }}
  ]
}}

YouTube Metadata Ozeti:
{metadata_ozeti}

B-Roll Ozeti:
{broll_ozeti}

Video Elestirmeni Ozeti:
{critic_ozeti}

Transcript:
{transkript}
""".strip()


def normalize_data(data: dict) -> dict:
    if not isinstance(data, dict):
        return {}

    candidates = []
    raw_candidates = data.get("story_candidates", [])
    if not isinstance(raw_candidates, list):
        raw_candidates = []

    for index, item in enumerate(raw_candidates, start=1):
        if not isinstance(item, dict):
            continue
        story_title_tr = normalize_whitespace(item.get("story_title_tr", ""))
        story_content_tr = normalize_whitespace(item.get("story_content_tr", item.get("story_direction_tr", "")))
        visual_direction_tr = normalize_whitespace(item.get("visual_direction_tr", ""))
        if not story_title_tr and not story_content_tr:
            continue
        candidates.append(
            {
                "rank": int(item.get("rank", index) or index),
                "engagement_score": max(0, min(100, int(item.get("engagement_score", item.get("score", 0)) or 0))),
                "story_title_tr": story_title_tr,
                "why_selected_tr": normalize_whitespace(item.get("why_selected_tr", "")),
                "story_content_tr": story_content_tr,
                "visual_direction_tr": visual_direction_tr,
                "interaction_tip_tr": normalize_whitespace(item.get("interaction_tip_tr", "")),
                **_normalize_story_prompt_fields(item),
            }
        )

    candidates.sort(key=lambda item: (-item.get("engagement_score", 0), item.get("rank", 999)))
    if len(candidates) > MAX_STORY_CANDIDATES:
        candidates = candidates[:MAX_STORY_CANDIDATES]
    for rank, item in enumerate(candidates, start=1):
        item["rank"] = rank

    if len(candidates) < MIN_STORY_CANDIDATES:
        logger.warning(f"Story adayi beklenenden az geldi: {len(candidates)}")

    return {
        "why_this_many_stories_tr": normalize_whitespace(data.get("why_this_many_stories_tr", "")),
        "selected_story_count": len(candidates),
        "story_candidates": candidates,
    }


def build_report_text(girdi_stem: str, data: dict, model_adi: str) -> str:
    lines = [
        f"=== {girdi_stem} ICIN INSTAGRAM STORY FIKIRLERI ===",
        f"Kullanilan Model: {model_adi}",
        "",
        "NEDEN BU KADAR STORY ADAYI SECILDI?",
        "-" * 60,
        data.get("why_this_many_stories_tr", ""),
        "",
        f"TOPLAM STORY ADAYI: {data.get('selected_story_count', 0)}",
        "",
    ]

    for item in data.get("story_candidates", []):
        lines.extend(
            [
                f"STORY #{item.get('rank', '')}",
                "-" * 60,
                f"Etkilesim Skoru: {item.get('engagement_score', 0)}/100",
                f"Story Basligi: {item.get('story_title_tr', '')}",
                f"Neden Secildi: {item.get('why_selected_tr', '')}",
                f"Storyde Ne Olmali?: {item.get('story_content_tr', '')}",
                f"Gorsel Nasil Olmali?: {item.get('visual_direction_tr', '')}",
                f"Etkilesim Taktigi: {item.get('interaction_tip_tr', '')}",
                "",
                "GORSEL URETIM PROMPTU (TR)",
                "-" * 40,
                build_story_image_prompt(item),
                "",
                "=" * 60,
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def _safe_file_fragment(value: object, default: str) -> str:
    normalized = normalize_whitespace(value)
    cleaned = re.sub(r'[<>:"/\\\\|?*]+', "", normalized).strip().strip(".")
    cleaned = re.sub(r"\s+", "-", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or default


def _build_single_story_text(girdi_stem: str, item: dict, model_adi: str) -> str:
    lines = [
        f"STORY #{item.get('rank', '')}",
        f"Kaynak: {girdi_stem}",
        f"Kullanilan Model: {model_adi}",
        "",
        f"Etkilesim Skoru: {item.get('engagement_score', 0)}/100",
        f"Story Basligi: {item.get('story_title_tr', '')}",
        f"Neden Secildi: {item.get('why_selected_tr', '')}",
        f"Storyde Ne Olmali?: {item.get('story_content_tr', '')}",
        f"Gorsel Nasil Olmali?: {item.get('visual_direction_tr', '')}",
        f"Etkilesim Taktigi: {item.get('interaction_tip_tr', '')}",
        "",
        "GORSEL URETIM PROMPTU (TR)",
        "-" * 40,
        build_story_image_prompt(item),
        "",
    ]
    return "\n".join(lines).strip() + "\n"


def _write_individual_story_txts(girdi_stem: str, data: dict, model_adi: str) -> None:
    if LEGACY_STORY_ITEM_TXT_DIR.exists():
        for old_file in LEGACY_STORY_ITEM_TXT_DIR.glob("*.txt"):
            old_file.unlink()
    STORY_ITEM_TXT_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in STORY_ITEM_TXT_DIR.glob("*.txt"):
        old_file.unlink()

    count = 0
    for item in data.get("story_candidates", []):
        rank = int(item.get("rank", count + 1) or (count + 1))
        title_fragment = _safe_file_fragment(item.get("story_title_tr", ""), "Baslik")
        path = STORY_ITEM_TXT_DIR / f"{rank:02d}-IG_Story-{title_fragment}.txt"
        path.write_text(_build_single_story_text(girdi_stem, item, model_adi), encoding="utf-8")
        count += 1

    logger.info(f"Tekil story TXT dosyalari kaydedildi: {count} adet")


def save_reports(girdi_dosyasi: Path, data: dict, model_adi: str) -> Tuple[Path, Optional[Path]]:
    json_yolu = stem_json_output_path(girdi_dosyasi.stem, "_instagram_story_plani.json", group="instagram")
    txt_yolu = txt_output_path("instagram_story")

    with open(json_yolu, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)

    txt_yolu.unlink(missing_ok=True)
    _write_individual_story_txts(girdi_dosyasi.stem, data, model_adi)
    logger.info(f"Instagram story plani JSON kaydedildi: {json_yolu.name}")
    logger.info(f"Tekil story TXT klasoru guncellendi: {STORY_ITEM_TXT_DIR.name}")
    return json_yolu, None


def analyze(
    girdi_dosyasi: Path,
    llm: CentralLLM,
    metadata_data: Optional[dict] = None,
    broll_data: Optional[list] = None,
    critic_data: Optional[dict] = None,
    draft_llm: Optional[CentralLLM] = None,
) -> dict:
    metadata_data = metadata_data or load_related_json(girdi_dosyasi, "_metadata.json")
    broll_data = broll_data or load_related_json(girdi_dosyasi, "_B_roll_fikirleri.json")
    critic_data = critic_data or load_related_json(girdi_dosyasi, "_video_critic.json")
    transkript = prepare_transcript(girdi_dosyasi, max_karakter=35000)
    metadata_ozeti = build_metadata_summary(metadata_data)
    broll_ozeti = build_broll_summary(broll_data)
    critic_ozeti = build_critic_summary(critic_data)
    draft_engine = draft_llm or llm

    logger.info(
        f"Instagram story fikir adaylari olusturuluyor... (minimum hedef: {MIN_STORY_CANDIDATES}, ust sinir: {MAX_STORY_CANDIDATES})"
    )
    parsed = None
    ideation = _request_llm(
        draft_engine,
        build_ideation_prompt(transkript, metadata_ozeti, broll_ozeti, critic_ozeti),
        profile="creative_ideation",
        debug_stem=girdi_dosyasi.stem,
        stage_label="ideation",
    )
    if isinstance(ideation, dict) and isinstance(ideation.get("story_candidates"), list):
        parsed = _request_llm(
            llm,
            build_selection_prompt(transkript, metadata_ozeti, broll_ozeti, critic_ozeti, ideation),
            profile="creative_ranker",
            debug_stem=girdi_dosyasi.stem,
            stage_label="selection",
        )
    if not parsed:
        parsed = _request_llm(
            llm,
            build_prompt(transkript, metadata_ozeti, broll_ozeti, critic_ozeti),
            profile="creative_ranker",
            debug_stem=girdi_dosyasi.stem,
            stage_label="fallback",
        )
    if not parsed:
        logger.error("Story planlayici cevabi parse edilemedi.")
        return {}
    normalized = normalize_data(parsed)
    if len(normalized.get("story_candidates", [])) < MIN_STORY_CANDIDATES and isinstance(ideation, dict):
        logger.warning(
            f"Story adayi minimum hedefin altinda kaldi ({len(normalized.get('story_candidates', []))}/{MIN_STORY_CANDIDATES}). Secim asamasi tekrar deneniyor..."
        )
        retried = _request_llm(
            llm,
            build_selection_prompt(transkript, metadata_ozeti, broll_ozeti, critic_ozeti, ideation),
            profile="creative_ranker",
            debug_stem=girdi_dosyasi.stem,
            stage_label="selection_retry",
        )
        if retried:
            retried_normalized = normalize_data(retried)
            if len(retried_normalized.get("story_candidates", [])) > len(normalized.get("story_candidates", [])):
                normalized = retried_normalized
    return normalized


def run():
    print("\n" + "=" * 60)
    print("STORY SERISI FIKIR URETICI")
    print("=" * 60)

    girdi = select_primary_srt(logger, "Story Serisi Fikir Uretici")
    if not girdi:
        return

    use_recommended = prompt_module_llm_plan("303", needs_main=True, needs_smart=True)
    if use_recommended:
        saglayici_ana, model_adi_ana = get_module_recommended_llm_config("303", "main")
        saglayici, model_adi = get_module_recommended_llm_config("303", "smart")
        print_module_llm_choice_summary(
            "303",
            {"main": (saglayici_ana, model_adi_ana), "smart": (saglayici, model_adi)},
        )
    else:
        saglayici_ana, model_adi_ana = select_llm("main")
        saglayici, model_adi = select_llm("smart")
    draft_llm = CentralLLM(provider=saglayici_ana, model_name=model_adi_ana)
    llm = CentralLLM(provider=saglayici, model_name=model_adi)

    data = analyze(girdi, llm, draft_llm=draft_llm)
    if not data:
        return logger.error("❌ Instagram story fikirleri uretilemedi.")

    save_reports(girdi, data, f"Draft: {model_adi_ana} | Final: {model_adi}")
    logger.info("🎉 Story serisi fikir uretimi tamamlandi.")


def run_automatic(
    girdi_dosyasi: Path,
    llm: CentralLLM,
    metadata_data: Optional[dict] = None,
    broll_data: Optional[list] = None,
    critic_data: Optional[dict] = None,
    draft_llm: Optional[CentralLLM] = None,
) -> Optional[dict]:
    logger.info(f"🔄 OTOMASYON: {girdi_dosyasi.name} icin Instagram story fikirleri uretiliyor...")
    data = analyze(
        girdi_dosyasi,
        llm,
        metadata_data=metadata_data,
        broll_data=broll_data,
        critic_data=critic_data,
        draft_llm=draft_llm,
    )
    if not data:
        logger.error("❌ Otomatik Instagram story fikir uretimi basarisiz oldu.")
        return None

    json_yolu, txt_yolu = save_reports(girdi_dosyasi, data, llm.model_name)
    return {
        "data": data,
        "json_path": json_yolu,
        "txt_path": txt_yolu,
    }
