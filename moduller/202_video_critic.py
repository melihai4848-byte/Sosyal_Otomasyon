import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from moduller.llm_manager import CentralLLM, print_module_llm_recommendation, select_llm
from moduller.logger import get_logger
from moduller.output_paths import stem_json_output_path, txt_output_path
from moduller.social_media_utils import select_primary_srt
from moduller.srt_utils import parse_srt_blocks, read_srt_file
from moduller.youtube_llm_profiles import call_with_youtube_profile

logger = get_logger("critic")


def normalize_whitespace(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def coerce_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "evet", "var", "gerekli", "run"}
    return False


def extract_json_response(llm_cevabi: str, logger_override=None, log_errors: bool = True):
    active_logger = logger_override or logger
    if not llm_cevabi:
        return None

    try:
        cevap = re.sub(r"```json\s*|```", "", str(llm_cevabi)).strip()

        try:
            return json.loads(cevap)
        except Exception:
            pass

        start_positions = [i for i, ch in enumerate(cevap) if ch == "{"]
        for start in start_positions:
            depth = 0
            in_string = False
            escape = False

            for idx in range(start, len(cevap)):
                ch = cevap[idx]

                if escape:
                    escape = False
                    continue

                if ch == "\\":
                    escape = True
                    continue

                if ch == '"':
                    in_string = not in_string
                    continue

                if not in_string:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            return json.loads(cevap[start:idx + 1])
    except Exception as e:
        if log_errors:
            active_logger.error(f"JSON ayrıştırma hatası: {e}")

    if log_errors:
        active_logger.error("LLM cevabından geçerli JSON çıkarılamadı.")
    return None


def prepare_srt_text(girdi_dosyasi: Path, max_karakter: int = 24000) -> str:
    icerik = read_srt_file(girdi_dosyasi)
    bloklar = parse_srt_blocks(icerik)
    satirlar = [f"[{b.timing_line}] {b.text_content}" for b in bloklar if b.is_processable]
    return "\n".join(satirlar)[:max_karakter]


def build_prompt(srt_metni: str) -> str:
    return f"""
Sen kıdemli bir YouTube içerik stratejisti, retention analisti, hikaye editörü ve acımasız ama yapıcı bir AI Critic'sin.

Görevin:
Aşağıdaki altyazı/transcript metnini analiz et ve videonun ne kadar güçlü olduğunu eleştirel biçimde değerlendir.
Kod tarafı bu analizden routing kararları türetecek; senin görevin mümkün olan en dürüst, açık ve uygulanabilir analiz verisini üretmek.

KRİTİK KURALLAR:
- Sadece geçerli JSON döndür.
- JSON dışında hiçbir açıklama yazma.
- Yumuşatma yapma; dürüst, net ve uygulanabilir geri bildirim ver.
- Kritikler yapıcı olsun; her zayıflığın yanında çözüm önerisi ver.
- Tüm metin alanları Türkçe olsun.
- Timestamp alanlarında transcriptte geçen gerçek zaman aralıklarını kullan.
- Uydurma detay ekleme; transcriptte olmayan şeyi kesinmiş gibi söyleme.
- Sayısal skorlar 0 ile 100 arasında olsun.
- En az 3 güçlü yön, en az 3 sorun, en az 5 aksiyon maddesi üret.

JSON ŞEMASI:
{{
  "overall_score": 0,
  "verdict": "",
  "target_audience_fit": "",
  "summary": "",
  "score_breakdown": {{
    "hook_strength": 0,
    "clarity": 0,
    "retention_potential": 0,
    "story_flow": 0,
    "practical_value": 0,
    "cta_strength": 0
  }},
  "strongest_points": ["", "", ""],
  "biggest_issues": ["", "", ""],
  "retention_analysis": {{
    "first_30_seconds": "",
    "middle_section": "",
    "ending": ""
  }},
  "timeline_notes": [
    {{
      "timestamp": "00:00:00,000 --> 00:00:15,000",
      "issue": "",
      "impact": "",
      "fix": ""
    }}
  ],
  "rewrite_opportunities": [
    {{
      "timestamp": "00:00:00,000 --> 00:00:15,000",
      "problem": "",
      "better_direction": ""
    }}
  ],
  "packaging_feedback": {{
    "title_thumbnail_potential": "",
    "shorts_clip_potential": "",
    "seo_topic_strength": ""
  }},
  "action_plan": ["", "", "", "", ""]
}}

Değerlendirme odağı:
- Açılış ilk saniyelerde yeterince kanca atıyor mu?
- Konu akışı dağınık mı, fazla genel mi, tekrar var mı?
- İzleyiciyi videoda tutacak sürpriz, gerilim, merak boşluğu var mı?
- Konuşma tarzı fazla düz, öğretici ama sıkıcı ya da fazla soyut mu?
- Son bölüm tatmin edici mi, güçlü bir kapanış/CTA var mı?
- Bu içerik uzun video ve Shorts açısından paketlenebilir mi?

Analiz edilecek transcript:
{srt_metni}
""".strip()


def _fallback_route_decisions(data: dict) -> Dict[str, dict]:
    breakdown = data.get("score_breakdown", {})
    packaging = data.get("packaging_feedback", {})

    blob = " ".join(
        [
            data.get("summary", ""),
            " ".join(data.get("biggest_issues", [])),
            data.get("retention_analysis", {}).get("first_30_seconds", ""),
            data.get("retention_analysis", {}).get("middle_section", ""),
            data.get("retention_analysis", {}).get("ending", ""),
            packaging.get("title_thumbnail_potential", ""),
            packaging.get("shorts_clip_potential", ""),
            packaging.get("seo_topic_strength", ""),
        ]
    ).lower()

    hook_strength = int(breakdown.get("hook_strength", 0) or 0)
    clarity = int(breakdown.get("clarity", 0) or 0)
    retention = int(breakdown.get("retention_potential", 0) or 0)
    story_flow = int(breakdown.get("story_flow", 0) or 0)
    practical_value = int(breakdown.get("practical_value", 0) or 0)
    overall = int(data.get("overall_score", 0) or 0)

    hook_run = hook_strength < 70 or "hook" in blob or "açılış" in blob or "ilk 30 saniye" in blob
    trim_run = (
        clarity < 70
        or story_flow < 70
        or any(token in blob for token in ["tekrar", "uzun", "dağınık", "gereksiz", "sarkıyor", "dolgu"])
    )
    broll_run = retention < 75 or any(token in blob for token in ["monoton", "görsel", "tekdüze", "durağan"])
    reels_run = any(token in blob for token in ["short", "reels", "kesilebilir", "viral", "clip", "klip"]) or overall >= 70
    metadata_run = (
        practical_value >= 60
        and clarity >= 60
        and not any(token in blob for token in ["konu net değil", "belirsiz", "çok dağınık"])
    )

    return {
        "hook_rewrite": {
            "run": hook_run,
            "reason": "Açılış bölümünde daha güçlü bir kanca ihtiyacı tespit edildi." if hook_run else "Açılış yeterince güçlü görünüyor."
        },
        "trim_suggestions": {
            "run": trim_run,
            "reason": "Akışta tekrar, dağınıklık veya gereksiz uzama ihtimali görüldü." if trim_run else "Akış yeterince temiz ve sıkı görünüyor."
        },
        "broll_generator": {
            "run": broll_run,
            "reason": "Retention riskini azaltmak için görsel destek faydalı olabilir." if broll_run else "Ek B-roll zorunlu görünmüyor."
        },
        "reels_shorts": {
            "run": reels_run,
            "reason": "İçerikte kısa formata kesilebilecek anlar var." if reels_run else "Shorts potansiyeli zayıf görünüyor."
        },
        "metadata_generator": {
            "run": metadata_run,
            "reason": "Konu, paketleme ve arama niyeti açısından metadata üretimi mantıklı." if metadata_run else "Önce içerik netliği güçlendirilmeli."
        },
    }


def normalize_route_decisions(data: dict) -> Dict[str, dict]:
    fallback = _fallback_route_decisions(data)
    incoming = data.get("routing_decisions", {})
    if not isinstance(incoming, dict):
        incoming = {}

    normalized = {}
    for key, default_value in fallback.items():
        raw = incoming.get(key, {})

        if isinstance(raw, dict):
            run_value = raw.get("run", default_value["run"])
            reason_value = normalize_whitespace(raw.get("reason", default_value["reason"]))
        else:
            run_value = raw
            reason_value = default_value["reason"]

        normalized[key] = {
            "run": coerce_to_bool(run_value),
            "reason": reason_value or default_value["reason"],
        }

    return normalized


def normalize_critic_data(data: dict) -> dict:
    if not isinstance(data, dict):
        return {}

    data.setdefault("overall_score", 0)
    data.setdefault("verdict", "")
    data.setdefault("target_audience_fit", "")
    data.setdefault("summary", "")
    data.setdefault("score_breakdown", {})
    data.setdefault("strongest_points", [])
    data.setdefault("biggest_issues", [])
    data.setdefault("retention_analysis", {})
    data.setdefault("timeline_notes", [])
    data.setdefault("rewrite_opportunities", [])
    data.setdefault("packaging_feedback", {})
    data.setdefault("routing_decisions", {})
    data.setdefault("action_plan", [])

    breakdown = data["score_breakdown"] if isinstance(data["score_breakdown"], dict) else {}
    for alan in [
        "hook_strength",
        "clarity",
        "retention_potential",
        "story_flow",
        "practical_value",
        "cta_strength",
    ]:
        try:
            breakdown[alan] = max(0, min(100, int(breakdown.get(alan, 0))))
        except Exception:
            breakdown[alan] = 0
    data["score_breakdown"] = breakdown

    try:
        data["overall_score"] = max(0, min(100, int(data.get("overall_score", 0))))
    except Exception:
        data["overall_score"] = 0

    for alan in ["strongest_points", "biggest_issues", "action_plan"]:
        if not isinstance(data.get(alan), list):
            data[alan] = []
        data[alan] = [normalize_whitespace(x) for x in data[alan] if normalize_whitespace(x)]

    if not isinstance(data.get("retention_analysis"), dict):
        data["retention_analysis"] = {}
    for alan in ["first_30_seconds", "middle_section", "ending"]:
        data["retention_analysis"][alan] = normalize_whitespace(data["retention_analysis"].get(alan, ""))

    if not isinstance(data.get("packaging_feedback"), dict):
        data["packaging_feedback"] = {}
    for alan in ["title_thumbnail_potential", "shorts_clip_potential", "seo_topic_strength"]:
        data["packaging_feedback"][alan] = normalize_whitespace(data["packaging_feedback"].get(alan, ""))

    temiz_timeline = []
    for item in data.get("timeline_notes", []):
        if not isinstance(item, dict):
            continue
        temiz_timeline.append({
            "timestamp": normalize_whitespace(item.get("timestamp", "")),
            "issue": normalize_whitespace(item.get("issue", "")),
            "impact": normalize_whitespace(item.get("impact", "")),
            "fix": normalize_whitespace(item.get("fix", "")),
        })
    data["timeline_notes"] = [x for x in temiz_timeline if x["issue"] or x["fix"]]

    temiz_rewrites = []
    for item in data.get("rewrite_opportunities", []):
        if not isinstance(item, dict):
            continue
        temiz_rewrites.append({
            "timestamp": normalize_whitespace(item.get("timestamp", "")),
            "problem": normalize_whitespace(item.get("problem", "")),
            "better_direction": normalize_whitespace(item.get("better_direction", "")),
        })
    data["rewrite_opportunities"] = [x for x in temiz_rewrites if x["problem"] or x["better_direction"]]

    data["routing_decisions"] = normalize_route_decisions(data)
    data["verdict"] = normalize_whitespace(data.get("verdict", ""))
    data["target_audience_fit"] = normalize_whitespace(data.get("target_audience_fit", ""))
    data["summary"] = normalize_whitespace(data.get("summary", ""))
    return data


def convert_report_to_printable_text(girdi_stem: str, data: dict, model_adi: str) -> str:
    breakdown = data.get("score_breakdown", {})
    retention = data.get("retention_analysis", {})
    packaging = data.get("packaging_feedback", {})
    timeline_notes = data.get("timeline_notes", [])

    lines = [
        f"=== {girdi_stem} ICIN POST-EDIT VIDEO ANALIZ RAPORU ===",
        f"Kullanilan Model: {model_adi}",
        "",
        f"GENEL SKOR: {data.get('overall_score', 0)}/100",
        f"KARAR: {data.get('verdict', '')}",
        f"HEDEF IZLEYICI UYUMU: {data.get('target_audience_fit', '')}",
        "",
        "OZET",
        "-" * 50,
        data.get("summary", ""),
        "",
        "SKOR KIRILIMI",
        "-" * 50,
        f"Hook Gucu: {breakdown.get('hook_strength', 0)}/100",
        f"Netlik: {breakdown.get('clarity', 0)}/100",
        f"Retention Potansiyeli: {breakdown.get('retention_potential', 0)}/100",
        f"Hikaye Akisi: {breakdown.get('story_flow', 0)}/100",
        f"Pratik Deger: {breakdown.get('practical_value', 0)}/100",
        f"CTA Gucu: {breakdown.get('cta_strength', 0)}/100",
        "",
        "EN GUCLU NOKTALAR",
        "-" * 50,
    ]

    for item in data.get("strongest_points", []):
        lines.append(f"- {item}")

    lines.extend([
        "",
        "EN BUYUK SORUNLAR",
        "-" * 50,
    ])
    for item in data.get("biggest_issues", []):
        lines.append(f"- {item}")

    lines.extend([
        "",
        "RETENTION ANALIZI",
        "-" * 50,
        f"Ilk 30 Saniye: {retention.get('first_30_seconds', '')}",
        f"Orta Bolum: {retention.get('middle_section', '')}",
        f"Kapanis: {retention.get('ending', '')}",
    ])

    if timeline_notes:
        lines.extend([
            "",
            "IZLEME SIRASINDA DIKKAT CEKEN ANLAR",
            "-" * 50,
        ])
        for item in timeline_notes:
            lines.append(f"Zaman: {item.get('timestamp', '')}")
            lines.append(f"Gozlem: {item.get('issue', '')}")
            lines.append(f"Etkisi: {item.get('impact', '')}")
            lines.append("")

    lines.extend([
        "",
        "PAKETLEME VE YAYIN POTANSIYELI",
        "-" * 50,
        f"Title/Thumbnail Potansiyeli: {packaging.get('title_thumbnail_potential', '')}",
        f"Shorts Klip Potansiyeli: {packaging.get('shorts_clip_potential', '')}",
        f"SEO Konu Gucu: {packaging.get('seo_topic_strength', '')}",
        "",
        "SONRAKI VIDEO ICIN DERSLER",
        "-" * 50,
    ])
    for item in data.get("action_plan", []):
        lines.append(f"- {item}")

    return "\n".join(lines).strip() + "\n"


def analyze(
    girdi_dosyasi: Path,
    llm: CentralLLM,
    prepared_transcript: Optional[str] = None,
) -> dict:
    srt_metni = normalize_whitespace(prepared_transcript) or prepare_srt_text(girdi_dosyasi)
    if not normalize_whitespace(srt_metni):
        logger.error("Analiz icin kullanilabilir SRT metni bulunamadi.")
        return {}

    prompt = build_prompt(srt_metni)
    logger.info("AI Critic transcripti analiz ediyor...")
    ham_cevap = call_with_youtube_profile(llm, prompt, profile="analytic_json")
    parsed = extract_json_response(ham_cevap, logger_override=logger)

    if not parsed:
        logger.error("AI Critic cevabi islenemedi.")
        return {}

    return normalize_critic_data(parsed)


def save_reports(girdi_dosyasi: Path, data: dict, model_adi: str) -> Tuple[Path, Path]:
    json_yolu = stem_json_output_path(girdi_dosyasi.stem, "_video_critic.json", group="youtube")
    txt_yolu = txt_output_path("video_critic")

    with open(json_yolu, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    txt_icerik = convert_report_to_printable_text(girdi_dosyasi.stem, data, model_adi)
    txt_yolu.write_text(txt_icerik, encoding="utf-8")

    logger.info(f"AI Critic JSON raporu kaydedildi: {json_yolu.name}")
    logger.info(f"AI Critic TXT raporu kaydedildi: {txt_yolu.name}")
    return json_yolu, txt_yolu


def run():
    print("\n" + "=" * 60)
    print("VİDEO ELEŞTİRMENİ | Video Eleştiri ve Routing Analizi")
    print("=" * 60)

    girdi = select_primary_srt(logger, "Video Elestirmeni")
    if not girdi:
        return

    print_module_llm_recommendation("202")
    saglayici, model_adi = select_llm("smart")
    llm = CentralLLM(provider=saglayici, model_name=model_adi)

    data = analyze(girdi, llm)
    if not data:
        return logger.error("❌ Video Critic raporu uretilemedi.")

    save_reports(girdi, data, model_adi)
    logger.info("🎉 AI Critic islemi tamamlandi.")


def run_automatic(
    girdi_dosyasi: Path,
    llm: CentralLLM,
    prepared_transcript: Optional[str] = None,
) -> Optional[dict]:
    logger.info(f"🔄 OTOMASYON: {girdi_dosyasi.name} icin AI Critic raporu uretiliyor...")

    data = analyze(girdi_dosyasi, llm, prepared_transcript=prepared_transcript)
    if not data:
        logger.error("❌ Otomatik AI Critic analizi basarisiz oldu.")
        return None

    json_yolu, txt_yolu = save_reports(girdi_dosyasi, data, llm.model_name)
    logger.info("🎉 Otomatik AI Critic raporu tamamlandi.")
    return {
        "data": data,
        "json_path": json_yolu,
        "txt_path": txt_yolu,
    }

