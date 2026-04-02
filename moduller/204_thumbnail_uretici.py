import json
import re
from pathlib import Path
from typing import Any, Optional

from moduller._module_alias import load_numbered_module
from moduller.llm_manager import (
    CentralLLM,
    get_module_recommended_llm_config,
    print_module_llm_choice_summary,
    prompt_module_llm_plan,
    select_llm,
)
from moduller.logger import get_logger
from moduller.output_paths import json_output_path, txt_output_path
from moduller.social_media_utils import (
    build_broll_summary,
    build_critic_summary,
    build_metadata_summary,
    load_related_json,
    select_metadata_language,
    select_primary_srt,
)
from moduller.srt_utils import parse_srt_blocks, read_srt_file
from moduller.youtube_llm_profiles import call_with_youtube_profile

_VIDEO_CRITIC_MODULE = load_numbered_module("202_video_critic.py")
coerce_to_bool = _VIDEO_CRITIC_MODULE.coerce_to_bool
extract_json_response = _VIDEO_CRITIC_MODULE.extract_json_response
normalize_whitespace = _VIDEO_CRITIC_MODULE.normalize_whitespace

YOUTUBE_LOGGER = get_logger("thumbnail_main")

HIGH_CONTRAST_SCORE = 25
MEDIUM_CONTRAST_SCORE = 16
LOW_CONTRAST_SCORE = 8
FACE_PRESENT_SCORE = 20
NO_FACE_SCORE = 8
YOUTUBE_MAIN_FORMAT_SPEC_TR = "Format/Cozunurluk: 16:9 (yatay) en-boy orani, onerilen cozunurluk 1920x1080."
YOUTUBE_MAIN_PROMPT_FOOTER_TR = (
    "Metin buyuk, mobilde rahat okunur olsun, 2-3 satiri gecmesin ve guclu gorsel hiyerarsiyle ana mesaja "
    "odaklansin."
)
TIMING_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})")
MAIN_VIDEO_TRANSCRIPT_MAX_CHARS = 12000
MAIN_VIDEO_TARGET_ANCHORS = 7
MAIN_VIDEO_WINDOW_RADIUS = 1


def _save_outputs(key: str, payload: dict, txt_icerik: str) -> tuple[Path, Path]:
    json_yolu = json_output_path(key)
    txt_yolu = txt_output_path(key)
    with open(json_yolu, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    txt_yolu.write_text(txt_icerik.strip() + "\n", encoding="utf-8")
    return json_yolu, txt_yolu


def _scoring_instructions(
    prompt_language_instruction: str = 'Aciklama alanlari Turkce, "prompt" alani ise sadece Ingilizce olsun.'
) -> str:
    return f"""
THUMBNAIL SCORING ENGINE:
- Her fikir icin CTR ihtimalini destekleyen analiz girdileri ekle.
- Ozellikle su 4 sinyal icin net deger ver:
  1) CTR tahmini dusuncesi
  2) Yuz var mi? / yok mu?
  3) Kontrast analizi
  4) Mobil okunabilirlik
- "mobile_readability_score" 0-100 arasi olsun.
- "visual_focus_score" 0-100 arasi olsun.
- "contrast_level" sadece "high", "medium" veya "low" olsun.
- "has_face" sadece true veya false olsun.
- {prompt_language_instruction}
""".strip()


def _timing_start_ms(timing_line: str) -> Optional[int]:
    if not timing_line:
        return None
    match = TIMING_RE.search(str(timing_line))
    if not match:
        return None
    hour, minute, second, milli = [int(x) for x in match.groups()]
    return (((hour * 60) + minute) * 60 + second) * 1000 + milli


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


def _extract_title_shortlist(metadata_data: Optional[dict], limit: int = 5) -> list[str]:
    secilen = select_metadata_language(metadata_data)
    if not secilen:
        return []
    sonuc = []
    gorulen = set()
    for item in secilen.get("titles", []):
        title = normalize_whitespace(item.get("title", "")) if isinstance(item, dict) else normalize_whitespace(item)
        if not title:
            continue
        anahtar = title.casefold()
        if anahtar in gorulen:
            continue
        gorulen.add(anahtar)
        sonuc.append(title)
        if len(sonuc) >= limit:
            break
    return sonuc


def _extract_reference_moments(
    metadata_data: Optional[dict],
    critic_data: Optional[dict],
    broll_data: Optional[list],
) -> list[int]:
    moments: list[int] = []

    secilen = select_metadata_language(metadata_data)
    if secilen:
        for item in secilen.get("chapters", [])[:6]:
            if not isinstance(item, dict):
                continue
            start_ms = _timing_start_ms(item.get("timestamp", ""))
            if start_ms is not None:
                moments.append(start_ms)

    if isinstance(critic_data, dict):
        for key in ("timeline_notes", "rewrite_opportunities"):
            for item in critic_data.get(key, [])[:4]:
                if not isinstance(item, dict):
                    continue
                start_ms = _timing_start_ms(item.get("timestamp", ""))
                if start_ms is not None:
                    moments.append(start_ms)

    if isinstance(broll_data, list):
        for item in broll_data[:5]:
            if not isinstance(item, dict):
                continue
            start_ms = _timing_start_ms(item.get("timestamp", ""))
            if start_ms is not None:
                moments.append(start_ms)

    return moments


def _build_adaptive_main_transcript(
    girdi_dosyasi: Path,
    max_chars: int = MAIN_VIDEO_TRANSCRIPT_MAX_CHARS,
    metadata_data: Optional[dict] = None,
    critic_data: Optional[dict] = None,
    broll_data: Optional[list] = None,
) -> str:
    blocks = [b for b in parse_srt_blocks(read_srt_file(girdi_dosyasi)) if b.is_processable]
    if not blocks:
        return ""

    lines = [f"[{b.timing_line}] {b.text_content}" for b in blocks]
    full_text = "\n".join(lines)
    if len(full_text) <= max_chars:
        return full_text

    anchor_indices = {0, max(0, len(blocks) - 1)}
    if len(blocks) > 1:
        anchor_indices.add(1)
        anchor_indices.add(max(0, len(blocks) - 2))
    if len(blocks) > 2:
        for idx in range(MAIN_VIDEO_TARGET_ANCHORS):
            fraction = idx / max(MAIN_VIDEO_TARGET_ANCHORS - 1, 1)
            anchor_indices.add(round((len(blocks) - 1) * fraction))

    start_times = [_timing_start_ms(b.timing_line or "") for b in blocks]
    for moment in _extract_reference_moments(metadata_data, critic_data, broll_data):
        closest = _find_closest_block_index(start_times, moment)
        if closest is not None:
            anchor_indices.add(closest)

    ordered_indices = []
    seen = set()
    for idx in sorted(anchor_indices):
        for expanded in range(max(0, idx - MAIN_VIDEO_WINDOW_RADIUS), min(len(blocks), idx + MAIN_VIDEO_WINDOW_RADIUS + 1)):
            if expanded in seen:
                continue
            seen.add(expanded)
            ordered_indices.append(expanded)

    selected_lines: list[str] = []
    current_length = 0
    for idx in ordered_indices:
        line = lines[idx]
        candidate_length = current_length + len(line) + (1 if selected_lines else 0)
        if candidate_length > max_chars and current_length >= int(max_chars * 0.72):
            break
        selected_lines.append(line)
        current_length = candidate_length

    if len(selected_lines) < 8:
        return full_text[:max_chars]
    return "\n".join(selected_lines)


def _build_strict_json_retry_prompt(prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        "ONEMLI JSON ZORLAMASI:\n"
        "- Yalnizca tek bir gecerli JSON nesnesi dondur.\n"
        "- Markdown, kod blogu, aciklama, brainstorm, on soz veya son soz ekleme.\n"
        "- JSON disinda hicbir sey yazma.\n"
    ).strip()


def _build_main_video_prompt(
    transkript: str,
    metadata_ozeti: str = "",
    critic_ozeti: str = "",
    broll_ozeti: str = "",
    title_shortlist: Optional[list[str]] = None,
) -> str:
    title_shortlist = title_shortlist or []
    title_lines = "\n".join(f"- {item}" for item in title_shortlist) or "- Ozel title shortlist yok."
    return f"""
Sen YouTube thumbnail packaging strategist'i ve AI image prompt engineer'sin.

Gorevin:
Asagidaki packaging sinyallerini ve temsili transcript kesitlerini kullanarak ana video icin 16:9 yatay formatta 5 farkli thumbnail fikri uret.

KRITIK KURALLAR:
- Once thumbnail CTR'sini en cok hangi duygu, karsitlik, risk, vaat veya gorsel metaforun yukseltecegini kisaca analiz edebilirsin.
- Analiz kismini yalnizca duz metin veya <brainstorm> etiketiyle ver; {{ }}, [ ] veya kod blogu kullanma.
- Analizden sonra cevabini tek bir gecerli JSON nesnesi ya da tek bir ```json``` blogu olarak bitir.
- Tum aciklama alanlari Turkce olsun.
- "prompt" alani sadece Ingilizce olsun.
- Title shortlist'teki ifadeleri aynen kopyalamak zorunda degilsin; gerekiyorsa daha sert, daha kisa, daha vurucu overlay sec.
- Metadata, AI Critic ve B-roll notlarini packaging ipucu gibi kullan; transcripti bunlarla birlikte yorumla.
- Her `prompt` alani tek parca kullanima hazir, katmanli ve YouTube thumbnail'e ozel bir gorsel uretim promptu olsun.
- Her `prompt` tam olarak su mantigi izlesin:
  Create a YouTube thumbnail concept.
  Goal: ...
  Background: ...
  Subject: ...
  Main Text: "..."
  Supporting Elements: ...
  Style/Mood: ...
  {YOUTUBE_MAIN_FORMAT_SPEC_TR}
  {YOUTUBE_MAIN_PROMPT_FOOTER_TR}
- Her prompt Nano Banana Pro 2, DALL-E veya benzeri image modellerine uygun olsun.
- Promptlar detayli, sinematik, YouTube CTR mantigina uygun ve birbirinden belirgin farkli olsun.
- "visual_description" alaninda thumbnailin ne gosterdigini Turkce ve net anlat.
- Overlay metinleri mobilde rahat okunacak kadar kisa tut.
- Uretilecek listeyi sen de en guclu CTR ihtimalinden zayif olana dogru dizmeye calis.
- Fikirler genel degil, videoya ozgu olsun.

{_scoring_instructions()}

JSON SEMASI:
{{
  "ideas": [
    {{
      "rank": 1,
      "concept": "",
      "visual_description": "",
      "overlay_text": "",
      "why_it_works": "",
      "prompt": "",
      "scoring_inputs": {{
        "has_face": false,
        "face_reason": "",
        "contrast_level": "high",
        "contrast_reason": "",
        "mobile_readability_score": 0,
        "mobile_readability_reason": "",
        "visual_focus_score": 0,
        "visual_focus_reason": ""
      }}
    }}
  ]
}}

METADATA OZETI:
{metadata_ozeti or "YouTube metadata verisi yok."}

AI CRITIC OZETI:
{critic_ozeti or "Video Elestirmeni verisi yok."}

B-ROLL MOTIFLERI:
{broll_ozeti or "B-roll verisi yok."}

TITLE SHORTLIST:
{title_lines}

Temsili Transcript Kesitleri:
{transkript}
""".strip()


def _bounded_int(value: Any, alt: int = 0, ust: int = 100, varsayilan: int = 0) -> int:
    try:
        return max(alt, min(ust, int(round(float(value)))))
    except Exception:
        return varsayilan


def _word_count(text: str) -> int:
    temiz = normalize_whitespace(text)
    if not temiz:
        return 0
    return len([parca for parca in temiz.replace("\n", " ").split(" ") if parca])


def _normalize_contrast_level(value: Any) -> str:
    text = normalize_whitespace(value).lower()
    if text in {"high", "yuksek", "yüksek"}:
        return "high"
    if text in {"medium", "orta"}:
        return "medium"
    if text in {"low", "dusuk", "düşük"}:
        return "low"
    return ""


def _infer_face_presence(item: dict, scoring_inputs: dict) -> bool:
    if "has_face" in scoring_inputs:
        return coerce_to_bool(scoring_inputs.get("has_face"))

    baglam = " ".join(
        [
            item.get("concept", ""),
            item.get("visual_description", ""),
            item.get("why_it_works", ""),
            item.get("visual_goal", ""),
            item.get("prompt", ""),
        ]
    ).lower()
    anahtarlar = [
        "face",
        "portrait",
        "close-up",
        "close up",
        "human",
        "person",
        "eyes",
        "expression",
        "smile",
        "smiling",
        "crying",
        "yuz",
        "yüz",
        "ifade",
        "insan",
        "kisi",
        "portre",
        "goz",
        "göz",
    ]
    return any(anahtar in baglam for anahtar in anahtarlar)


def _infer_contrast_level(item: dict, scoring_inputs: dict) -> str:
    normalized = _normalize_contrast_level(scoring_inputs.get("contrast_level", ""))
    if normalized:
        return normalized

    baglam = " ".join(
        [
            item.get("visual_description", ""),
            item.get("why_it_works", ""),
            item.get("visual_goal", ""),
            item.get("prompt", ""),
        ]
    ).lower()
    yuksek = [
        "high contrast",
        "dramatic lighting",
        "bold shadows",
        "neon",
        "spotlight",
        "rim light",
        "dark background",
        "bright highlight",
        "strong contrast",
        "yuksek kontrast",
        "dramatik isik",
        "guclu golge",
        "spot isik",
        "karanlik arka plan",
        "parlak vurgu",
        "sert kontrast",
    ]
    orta = [
        "soft contrast",
        "balanced lighting",
        "clean light",
        "natural light",
        "yumusak kontrast",
        "dengeli aydinlatma",
        "temiz isik",
        "dogal isik",
    ]

    if any(anahtar in baglam for anahtar in yuksek):
        return "high"
    if any(anahtar in baglam for anahtar in orta):
        return "medium"
    return "medium"


def _infer_mobile_readability(overlay_text: str, scoring_inputs: dict) -> int:
    score = _bounded_int(scoring_inputs.get("mobile_readability_score"), varsayilan=-1)
    if score >= 0:
        return score

    kelime_sayisi = _word_count(overlay_text)
    karakter_sayisi = len(normalize_whitespace(overlay_text))

    if kelime_sayisi == 0:
        base = 74
    elif kelime_sayisi <= 3:
        base = 92
    elif kelime_sayisi <= 5:
        base = 84
    elif kelime_sayisi <= 7:
        base = 72
    elif kelime_sayisi <= 10:
        base = 58
    else:
        base = 42

    if karakter_sayisi > 28:
        base -= 8
    elif karakter_sayisi > 20:
        base -= 4

    return max(20, min(95, base))


def _infer_visual_focus(item: dict, scoring_inputs: dict) -> int:
    score = _bounded_int(scoring_inputs.get("visual_focus_score"), varsayilan=-1)
    if score >= 0:
        return score

    prompt = item.get("prompt", "").lower()
    bonus = 0
    if any(
        token in prompt
        for token in [
            "single subject",
            "close-up",
            "tight framing",
            "center composition",
            "tek ozne",
            "yakin plan",
            "siki kadraj",
            "merkez kompozisyon",
        ]
    ):
        bonus += 12
    if any(
        token in prompt
        for token in [
            "crowd",
            "multiple people",
            "busy background",
            "wide shot",
            "kalabalik",
            "birden fazla kisi",
            "karisik arka plan",
            "genis plan",
        ]
    ):
        bonus -= 12
    return max(35, min(90, 70 + bonus))


def _text_density_score(overlay_text: str) -> tuple[int, str]:
    kelime_sayisi = _word_count(overlay_text)
    if kelime_sayisi == 0:
        return 10, "Overlay metni yok; sade tasarim avantajli olabilir ama mesaj netligi kontrol edilmeli."
    if kelime_sayisi <= 4:
        return 15, "Overlay metni kisa oldugu icin mobil ekranda daha rahat okunur."
    if kelime_sayisi <= 6:
        return 11, "Overlay metni makul uzunlukta; mobil okunabilirlik hala guclu."
    if kelime_sayisi <= 8:
        return 7, "Overlay metni biraz kalabalik; mobilde dikkat dagitabilir."
    return 3, "Overlay metni uzun; mobil CTR potansiyelini dusurebilir."


def _ctr_band(score: int) -> str:
    if score >= 82:
        return "Yuksek Potansiyel"
    if score >= 68:
        return "Orta Potansiyel"
    return "Daha Riskli"


def _build_thumbnail_scoring_engine(item: dict) -> dict:
    scoring_inputs = item.get("scoring_inputs", {}) if isinstance(item.get("scoring_inputs"), dict) else {}
    overlay_text = normalize_whitespace(item.get("overlay_text", item.get("title", "")))

    has_face = _infer_face_presence(item, scoring_inputs)
    contrast_level = _infer_contrast_level(item, scoring_inputs)
    mobile_readability = _infer_mobile_readability(overlay_text, scoring_inputs)
    visual_focus = _infer_visual_focus(item, scoring_inputs)

    face_score = FACE_PRESENT_SCORE if has_face else NO_FACE_SCORE
    contrast_score = {
        "high": HIGH_CONTRAST_SCORE,
        "medium": MEDIUM_CONTRAST_SCORE,
        "low": LOW_CONTRAST_SCORE,
    }.get(contrast_level, MEDIUM_CONTRAST_SCORE)
    mobile_weighted_score = round(mobile_readability * 0.22, 1)
    visual_focus_weighted_score = round(visual_focus * 0.17, 1)
    text_density_score, text_density_reason = _text_density_score(overlay_text)

    ctr_estimate = round(
        face_score
        + contrast_score
        + mobile_weighted_score
        + visual_focus_weighted_score
        + text_density_score
    )
    ctr_estimate = max(1, min(99, ctr_estimate))

    face_reason = normalize_whitespace(scoring_inputs.get("face_reason", ""))
    if not face_reason:
        face_reason = (
            "Insan yuzu/ifadesi var; duygusal bag kurma ve dikkat cekme ihtimali artar."
            if has_face
            else "Belirgin bir yuz odagi yok; CTR tamamen obje, tipografi ve kompozisyona yaslanir."
        )

    contrast_reason = normalize_whitespace(scoring_inputs.get("contrast_reason", ""))
    if not contrast_reason:
        contrast_reason = {
            "high": "Yuksek kontrast, feed icinde hizli fark edilme sansini artirir.",
            "medium": "Kontrast dengeli; guvenli ama daha az agresif bir dikkat cekme seviyesi sunar.",
            "low": "Dusuk kontrast, feed icinde ayirt edilmeyi zorlastirabilir.",
        }[contrast_level]

    mobile_reason = normalize_whitespace(scoring_inputs.get("mobile_readability_reason", ""))
    if not mobile_reason:
        mobile_reason = (
            f"Mobil okunabilirlik, overlay uzunlugu ve sadelik baz alinarak {mobile_readability}/100 olarak degerlendirildi."
        )

    visual_focus_reason = normalize_whitespace(scoring_inputs.get("visual_focus_reason", ""))
    if not visual_focus_reason:
        visual_focus_reason = (
            f"Gorsel odak netligi {visual_focus}/100 seviyesinde; ana mesajin tek karede anlasilabilirligi buna gore hesaplandi."
        )

    return {
        "ctr_estimate": ctr_estimate,
        "ctr_band": _ctr_band(ctr_estimate),
        "face_present": has_face,
        "face_analysis": {
            "score": face_score,
            "reason": face_reason,
        },
        "contrast_analysis": {
            "level": contrast_level,
            "score": contrast_score,
            "reason": contrast_reason,
        },
        "mobile_readability": {
            "score": mobile_readability,
            "weighted_score": mobile_weighted_score,
            "reason": mobile_reason,
        },
        "visual_focus": {
            "score": visual_focus,
            "weighted_score": visual_focus_weighted_score,
            "reason": visual_focus_reason,
        },
        "text_density": {
            "overlay_word_count": _word_count(overlay_text),
            "score": text_density_score,
            "reason": text_density_reason,
        },
        "analysis_summary": (
            f"CTR tahmini {ctr_estimate}/100. "
            f"Yuz {'var' if has_face else 'yok'}, kontrast {contrast_level}, "
            f"mobil okunabilirlik {mobile_readability}/100."
        ),
    }


def _normalize_idea(item: dict, fallback_rank: int) -> Optional[dict]:
    if not isinstance(item, dict):
        return None

    normalized = {
        "rank": _bounded_int(item.get("rank", fallback_rank), alt=1, ust=999, varsayilan=fallback_rank),
        "concept": normalize_whitespace(item.get("concept", "")),
        "visual_description": normalize_whitespace(item.get("visual_description", "")),
        "overlay_text": normalize_whitespace(item.get("overlay_text", "")),
        "why_it_works": normalize_whitespace(item.get("why_it_works", "")),
        "prompt": normalize_whitespace(item.get("prompt", "")),
        "scoring_inputs": item.get("scoring_inputs", {}) if isinstance(item.get("scoring_inputs"), dict) else {},
    }

    if not any([normalized["concept"], normalized["visual_description"], normalized["prompt"]]):
        return None

    normalized["thumbnail_scoring_engine"] = _build_thumbnail_scoring_engine(normalized)
    return normalized


def _sort_ideas_by_ctr(ideas: list[dict]) -> list[dict]:
    ordered = sorted(
        ideas,
        key=lambda item: (
            -item.get("thumbnail_scoring_engine", {}).get("ctr_estimate", 0),
            -item.get("thumbnail_scoring_engine", {}).get("mobile_readability", {}).get("score", 0),
            item.get("rank", 999),
        ),
    )
    for index, idea in enumerate(ordered, start=1):
        idea["rank"] = index
    return ordered


def _render_scoring_lines(item: dict) -> list[str]:
    scoring = item.get("thumbnail_scoring_engine", {})
    contrast = scoring.get("contrast_analysis", {})
    mobile = scoring.get("mobile_readability", {})
    focus = scoring.get("visual_focus", {})
    text_density = scoring.get("text_density", {})

    return [
        f"CTR Tahmini: {scoring.get('ctr_estimate', 0)}/100 ({scoring.get('ctr_band', '')})",
        f"Yuz Var Mi?: {'Evet' if scoring.get('face_present') else 'Hayir'}",
        f"Yuz Analizi: {scoring.get('face_analysis', {}).get('reason', '')}",
        f"Kontrast Analizi: {contrast.get('level', '')} | {contrast.get('reason', '')}",
        f"Mobil Okunabilirlik: {mobile.get('score', 0)}/100 | {mobile.get('reason', '')}",
        f"Gorsel Odak: {focus.get('score', 0)}/100 | {focus.get('reason', '')}",
        f"Metin Yogunlugu: {text_density.get('overlay_word_count', 0)} kelime | {text_density.get('reason', '')}",
        f"Skor Ozeti: {scoring.get('analysis_summary', '')}",
    ]


def _ideas_to_txt(title: str, items: list[dict], model_adi: str = "") -> str:
    lines = [title]
    if model_adi:
        lines.append(f"Kullanilan Model: {model_adi}")
    lines.append("")
    for item in items:
        sira = item.get("rank", "")
        baslik = item.get("concept", "")
        lines.extend(
            [
                f"#{sira} | {baslik}",
                "-" * 60,
            ]
        )

        lines.extend(
            [
                f"Turkce Gorsel Aciklama: {item.get('visual_description', '')}",
                f"Overlay / Baslik: {item.get('overlay_text', '')}",
                f"Neden ise yarar: {item.get('why_it_works', '')}",
            ]
        )
        lines.extend(_render_scoring_lines(item))
        lines.extend(["Gorsel Uretim Promptu:", item.get("prompt", ""), ""])

    return "\n".join(lines).strip()


def _select_srt() -> Optional[Path]:
    return select_primary_srt(YOUTUBE_LOGGER, "Thumbnail Prompt Uretici")


def _generate_thumbnail_payload(llm: CentralLLM, prompt: str, active_logger) -> dict:
    parsed = extract_json_response(
        call_with_youtube_profile(llm, prompt, profile="creative_ideation"),
        logger_override=active_logger,
    )
    if isinstance(parsed, dict) and (
        (isinstance(parsed.get("ideas"), list) and parsed.get("ideas"))
        or (isinstance(parsed.get("slides"), list) and parsed.get("slides"))
    ):
        return parsed

    active_logger.warning("Thumbnail creative_ideation cevabi parse edilemedi veya bos geldi. strict_json fallback deneniyor.")
    strict_retry = extract_json_response(
        call_with_youtube_profile(llm, _build_strict_json_retry_prompt(prompt), profile="strict_json"),
        logger_override=active_logger,
    )
    return strict_retry if isinstance(strict_retry, dict) else {}


def _prepare_llm(module_number: str) -> CentralLLM:
    use_recommended = prompt_module_llm_plan(module_number, needs_smart=True)
    if use_recommended:
        saglayici, model_adi = get_module_recommended_llm_config(module_number, "smart")
        print_module_llm_choice_summary(module_number, {"smart": (saglayici, model_adi)})
    else:
        saglayici, model_adi = select_llm("smart")
    return CentralLLM(provider=saglayici, model_name=model_adi)


def create_main_video_thumbnails(
    girdi_dosyasi: Path,
    llm: CentralLLM,
    prepared_transcript: Optional[str] = None,
    metadata_data: Optional[dict] = None,
    critic_data: Optional[dict] = None,
    broll_data: Optional[list] = None,
) -> Optional[dict]:
    if metadata_data is None:
        metadata_data = load_related_json(girdi_dosyasi, "_metadata.json")
    if critic_data is None:
        critic_data = load_related_json(girdi_dosyasi, "_video_critic.json")
    if broll_data is None:
        broll_data = load_related_json(girdi_dosyasi, "_B_roll_fikirleri.json")

    transkript = normalize_whitespace(prepared_transcript) if prepared_transcript else ""
    if transkript:
        transkript = prepared_transcript.strip()[:MAIN_VIDEO_TRANSCRIPT_MAX_CHARS]
    else:
        transkript = _build_adaptive_main_transcript(
            girdi_dosyasi,
            max_chars=MAIN_VIDEO_TRANSCRIPT_MAX_CHARS,
            metadata_data=metadata_data,
            critic_data=critic_data,
            broll_data=broll_data if isinstance(broll_data, list) else None,
        )
    if not normalize_whitespace(transkript):
        return None

    metadata_ozeti = build_metadata_summary(metadata_data)
    critic_ozeti = build_critic_summary(critic_data)
    broll_ozeti = build_broll_summary(broll_data if isinstance(broll_data, list) else None)
    title_shortlist = _extract_title_shortlist(metadata_data)

    data = _generate_thumbnail_payload(
        llm,
        _build_main_video_prompt(
            transkript,
            metadata_ozeti=metadata_ozeti,
            critic_ozeti=critic_ozeti,
            broll_ozeti=broll_ozeti,
            title_shortlist=title_shortlist,
        ),
        YOUTUBE_LOGGER,
    )
    ham_ideas = data.get("ideas", []) if isinstance(data.get("ideas"), list) else []
    ideas = [_normalize_idea(item, index) for index, item in enumerate(ham_ideas, start=1)]
    ideas = [item for item in ideas if item]
    ideas = _sort_ideas_by_ctr(ideas)[:5]
    if not ideas:
        YOUTUBE_LOGGER.warning("Ana video thumbnail cikti listesi bos geldi.")
        return None

    payload = {
        "source_srt": girdi_dosyasi.name,
        "model_name": llm.model_name,
        "context": {
            "metadata_loaded": isinstance(metadata_data, dict),
            "critic_loaded": isinstance(critic_data, dict),
            "broll_loaded": isinstance(broll_data, list),
            "title_shortlist": title_shortlist,
        },
        "ideas": ideas,
    }
    txt = _ideas_to_txt("=== ANA VIDEO THUMBNAIL FIKIRLERI ===", payload["ideas"], llm.model_name)
    json_yolu, txt_yolu = _save_outputs("main_video_thumbnails", payload, txt)
    return {"data": payload, "json_path": json_yolu, "txt_path": txt_yolu}


def run_youtube_thumbnail():
    print("\n" + "=" * 60)
    print("THUMBNAIL PROMPT URETICI | 16:9 YATAY")
    print("=" * 60)

    girdi = _select_srt()
    if not girdi:
        return

    llm = _prepare_llm("204")
    result = create_main_video_thumbnails(
        girdi,
        llm,
        metadata_data=load_related_json(girdi, "_metadata.json"),
        critic_data=load_related_json(girdi, "_video_critic.json"),
        broll_data=load_related_json(girdi, "_B_roll_fikirleri.json"),
    )
    if not result:
        YOUTUBE_LOGGER.error("❌ 16:9 thumbnail promptlari uretilemedi.")
        return

    print("\n🎉 16:9 thumbnail promptlari kaydedildi:")
    print(f"JSON: {result['json_path']}")
    print(f"TXT:  {result['txt_path']}")
def run():
    run_youtube_thumbnail()


