import json
from pathlib import Path
from typing import Optional, Tuple

from moduller._module_alias import load_numbered_module
from moduller.llm_manager import CentralLLM, select_llm
from moduller.logger import get_logger
from moduller.output_paths import stem_json_output_path, txt_output_path
from moduller.social_media_utils import select_primary_srt
from moduller.srt_utils import parse_srt_blocks, read_srt_file
from moduller.youtube_llm_profiles import call_with_youtube_profile

_ANALYTICS_FEEDBACK_MODULE = load_numbered_module("402_analitik_geri_bildirim_dongusu.py")
load_latest_feedback_data = _ANALYTICS_FEEDBACK_MODULE.load_latest_feedback_data
build_feedback_summary = _ANALYTICS_FEEDBACK_MODULE.build_feedback_summary

_VIDEO_CRITIC_MODULE = load_numbered_module("202_video_critic.py")
extract_json_response = _VIDEO_CRITIC_MODULE.extract_json_response
normalize_whitespace = _VIDEO_CRITIC_MODULE.normalize_whitespace

logger = get_logger("trim")


def prepare_transcript(girdi_dosyasi: Path, max_karakter: int = 22000) -> str:
    icerik = read_srt_file(girdi_dosyasi)
    bloklar = parse_srt_blocks(icerik)
    satirlar = [f"[{b.timing_line}] {b.text_content}" for b in bloklar if b.is_processable]
    return "\n".join(satirlar)[:max_karakter]


def build_critic_summary(critic_data: Optional[dict]) -> str:
    if not isinstance(critic_data, dict):
        return "AI Critic verisi yok. Yalnizca transcript akisina gore trim firsatlari cikar."

    return (
        f"Ozet: {critic_data.get('summary', '')}\n"
        f"Netlik skoru: {critic_data.get('score_breakdown', {}).get('clarity', 0)}\n"
        f"Hikaye akisi skoru: {critic_data.get('score_breakdown', {}).get('story_flow', 0)}\n"
        f"Buyuk sorunlar: {' | '.join(critic_data.get('biggest_issues', []))}"
    ).strip()


def build_prompt(transcript: str, critic_ozeti: str, feedback_ozeti: str) -> str:
    return f"""
Sen ust duzey bir YouTube editoru, retention kurgucusu ve trim strategist'sin.

Gorevin:
Asagidaki transcriptte izleyiciyi dusurebilecek tekrar, dolgu, gevsek anlatim veya gereksiz uzama noktalarini bul.

KRITIK KURALLAR:
- Sadece gecerli JSON dondur.
- JSON disinda hicbir aciklama yazma.
- Tum ana metinler Turkce olsun.
- Timestamp alanlarinda yalnizca transcriptte bulunan zaman araliklarini kullan.
- Gercekten kesmeye degecek bolumleri sec; rastgele trim onerme.
- En az 3 trim hedefi ve en az 5 aksiyon maddesi uret.
- Cok Kritik: Eger transcript bir Vlog'a (seyahat, eglence vs.) aitse, İLK 30 SANİYEDE bulunan birbiriyle baglantisiz, kopuk veya hizli gecisli diyaloglari SAKIN kesilecek (TRIM) hedefi olarak gösterme! Bu kisimlar videonun "Fast Teaser (Hizli Kanca)" montajidir ve kalmalari gerekir.
- Cok Kritik: Seyahat/Vlog videolarindaki altyazisiz 2-5 saniyelik bosluklari B-Roll olarak kabul et, "gereksiz sessizlik" diye kesit onerme.

JSON SEMASI:
{{
  "summary": "",
  "estimated_retention_gain": "",
  "trim_targets": [
    {{
      "timestamp": "00:00:00,000 --> 00:00:15,000",
      "problem": "",
      "trim_type": "",
      "reason": "",
      "suggested_action": "",
      "rewrite_example": ""
    }}
  ],
  "priority_cuts": ["", "", ""],
  "keep_as_is": ["", "", ""],
  "action_plan": ["", "", "", "", ""]
}}

AI Critic Ozeti:
{critic_ozeti}

Retention Geri Bildirim Ozeti:
{feedback_ozeti}

Transcript:
{transcript}
""".strip()


def normalize_data(data: dict) -> dict:
    if not isinstance(data, dict):
        return {}

    data.setdefault("summary", "")
    data.setdefault("estimated_retention_gain", "")
    data.setdefault("trim_targets", [])
    data.setdefault("priority_cuts", [])
    data.setdefault("keep_as_is", [])
    data.setdefault("action_plan", [])
    data.setdefault("skipped_by_routing", False)
    data.setdefault("routing_reason", "")

    data["summary"] = normalize_whitespace(data.get("summary", ""))
    data["estimated_retention_gain"] = normalize_whitespace(data.get("estimated_retention_gain", ""))
    data["priority_cuts"] = [normalize_whitespace(x) for x in data.get("priority_cuts", []) if normalize_whitespace(x)]
    data["keep_as_is"] = [normalize_whitespace(x) for x in data.get("keep_as_is", []) if normalize_whitespace(x)]
    data["action_plan"] = [normalize_whitespace(x) for x in data.get("action_plan", []) if normalize_whitespace(x)]
    data["routing_reason"] = normalize_whitespace(data.get("routing_reason", ""))
    data["skipped_by_routing"] = bool(data.get("skipped_by_routing"))

    temiz_targets = []
    for item in data.get("trim_targets", []):
        if not isinstance(item, dict):
            continue
        hedef = {
            "timestamp": normalize_whitespace(item.get("timestamp", "")),
            "problem": normalize_whitespace(item.get("problem", "")),
            "trim_type": normalize_whitespace(item.get("trim_type", "")),
            "reason": normalize_whitespace(item.get("reason", "")),
            "suggested_action": normalize_whitespace(item.get("suggested_action", "")),
            "rewrite_example": normalize_whitespace(item.get("rewrite_example", "")),
        }
        if hedef["problem"] or hedef["suggested_action"]:
            temiz_targets.append(hedef)

    data["trim_targets"] = temiz_targets
    return data


def _routing_decision(critic_data: Optional[dict]) -> Tuple[Optional[bool], str]:
    if not isinstance(critic_data, dict):
        return None, ""

    routing = critic_data.get("routing_decisions", {})
    if not isinstance(routing, dict):
        return None, ""

    item = routing.get("trim_suggestions", {})
    if not isinstance(item, dict):
        return None, ""

    if "run" not in item:
        return None, normalize_whitespace(item.get("reason", ""))

    return bool(item.get("run")), normalize_whitespace(item.get("reason", ""))


def _build_skipped_data(reason: str, critic_data: Optional[dict]) -> dict:
    note = normalize_whitespace(reason) or "AI Critic su an kapsamli bir trim ihtiyaci gormuyor."
    keep_as_is = []
    if isinstance(critic_data, dict):
        keep_as_is = [normalize_whitespace(x) for x in critic_data.get("strongest_points", []) if normalize_whitespace(x)][:3]

    return normalize_data(
        {
            "summary": note,
            "estimated_retention_gain": "Daha derin trim calismasi zorunlu gorulmedi.",
            "trim_targets": [],
            "priority_cuts": [],
            "keep_as_is": keep_as_is,
            "action_plan": [
                "Mevcut akisi koru ve sadece bariz tekrar gorursen manuel temizle.",
                "Bu rapor maliyeti azaltmak icin AI Critic routing kararina gore hafif modda olusturuldu.",
            ],
            "skipped_by_routing": True,
            "routing_reason": note,
        }
    )


def build_report_text(girdi_stem: str, data: dict, model_adi: str) -> str:
    if data.get("skipped_by_routing"):
        lines = [
            f"=== {girdi_stem} ICIN TRIM SUGGESTION RAPORU ===",
            f"Kullanilan Model: {model_adi}",
            "",
            "ROUTING KARARI",
            "-" * 50,
            data.get("routing_reason", ""),
            "",
            "KORUNMASI GEREKEN BOLUMLER",
            "-" * 50,
        ]
        for item in data.get("keep_as_is", []):
            lines.append(f"- {item}")
        lines.extend(["", "AKSIYON PLANI", "-" * 50])
        for item in data.get("action_plan", []):
            lines.append(f"- {item}")
        return "\n".join(lines).strip() + "\n"

    lines = [
        f"=== {girdi_stem} ICIN TRIM SUGGESTION RAPORU ===",
        f"Kullanilan Model: {model_adi}",
        "",
        "OZET",
        "-" * 50,
        data.get("summary", ""),
        "",
        "TAHMINI RETENTION KAZANIMI",
        "-" * 50,
        data.get("estimated_retention_gain", ""),
        "",
        "TRIM HEDEFLERI",
        "-" * 50,
    ]

    for idx, item in enumerate(data.get("trim_targets", []), 1):
        lines.append(f"Hedef {idx} | {item.get('timestamp', '')}")
        lines.append(f"Problem: {item.get('problem', '')}")
        lines.append(f"Trim tipi: {item.get('trim_type', '')}")
        lines.append(f"Neden: {item.get('reason', '')}")
        lines.append(f"Onerilen aksiyon: {item.get('suggested_action', '')}")
        lines.append(f"Rewrite ornegi: {item.get('rewrite_example', '')}")
        lines.append("")

    lines.extend([
        "ONCELIKLI KESILER",
        "-" * 50,
    ])
    for item in data.get("priority_cuts", []):
        lines.append(f"- {item}")

    lines.extend([
        "",
        "KORUNMASI GEREKEN BOLUMLER",
        "-" * 50,
    ])
    for item in data.get("keep_as_is", []):
        lines.append(f"- {item}")

    lines.extend([
        "",
        "AKSIYON PLANI",
        "-" * 50,
    ])
    for item in data.get("action_plan", []):
        lines.append(f"- {item}")

    return "\n".join(lines).strip() + "\n"


def analyze(
    girdi_dosyasi: Path,
    llm: CentralLLM,
    critic_data: Optional[dict] = None,
    feedback_data: Optional[dict] = None,
    prepared_transcript: Optional[str] = None,
    critic_summary: Optional[str] = None,
    feedback_summary: Optional[str] = None,
    respect_routing: bool = False,
) -> dict:
    transcript = normalize_whitespace(prepared_transcript) or prepare_transcript(girdi_dosyasi)
    if not normalize_whitespace(transcript):
        logger.error("Trim suggestion icin transcript bulunamadi.")
        return {}

    routing_run, routing_reason = _routing_decision(critic_data)
    if respect_routing and routing_run is False:
        logger.info("AI Critic trim adimini gereksiz gordu; hafif skip raporu olusturuluyor.")
        return _build_skipped_data(routing_reason, critic_data)

    feedback_data = feedback_data or load_latest_feedback_data()
    critic_ozeti = normalize_whitespace(critic_summary) or build_critic_summary(critic_data)
    feedback_ozeti = normalize_whitespace(feedback_summary) or build_feedback_summary(feedback_data)
    prompt = build_prompt(transcript, critic_ozeti, feedback_ozeti)
    logger.info("Trim Suggester tekrar ve gevsek bolumleri ariyor...")
    parsed = extract_json_response(
        call_with_youtube_profile(llm, prompt, profile="analytic_json"),
        logger_override=logger,
    )
    if not parsed:
        logger.error("Trim Suggester cevabi parse edilemedi.")
        return {}
    return normalize_data(parsed)


def save_reports(girdi_dosyasi: Path, data: dict, model_adi: str) -> Tuple[Path, Path]:
    json_yolu = stem_json_output_path(girdi_dosyasi.stem, "_trim_suggestions.json", group="youtube")
    txt_yolu = txt_output_path("trim_suggestions")

    with open(json_yolu, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    txt_yolu.write_text(build_report_text(girdi_dosyasi.stem, data, model_adi), encoding="utf-8")
    logger.info(f"Trim suggestion JSON kaydedildi: {json_yolu.name}")
    logger.info(f"Trim suggestion TXT kaydedildi: {txt_yolu.name}")
    return json_yolu, txt_yolu


def run():
    print("\n" + "=" * 60)
    print("VIDEODA KESİLECEK/ATILACAK YER ÖNERİCİ")
    print("=" * 60)

    girdi = select_primary_srt(logger, "Trim Onerici")
    if not girdi:
        return

    saglayici, model_adi = select_llm("main")
    llm = CentralLLM(provider=saglayici, model_name=model_adi)

    data = analyze(girdi, llm)
    if not data:
        return logger.error("❌ Trim suggestion uretilemedi.")

    save_reports(girdi, data, model_adi)
    logger.info("🎉 Trim Suggester islemi tamamlandi.")


def run_automatic(
    girdi_dosyasi: Path,
    llm: CentralLLM,
    critic_data: Optional[dict] = None,
    feedback_data: Optional[dict] = None,
    prepared_transcript: Optional[str] = None,
    critic_summary: Optional[str] = None,
    feedback_summary: Optional[str] = None,
    respect_routing: bool = False,
) -> Optional[dict]:
    logger.info(f"🔄 OTOMASYON: {girdi_dosyasi.name} icin trim suggestions uretiliyor...")
    data = analyze(
        girdi_dosyasi,
        llm,
        critic_data=critic_data,
        feedback_data=feedback_data,
        prepared_transcript=prepared_transcript,
        critic_summary=critic_summary,
        feedback_summary=feedback_summary,
        respect_routing=respect_routing,
    )
    if not data:
        logger.error("❌ Otomatik trim suggestion basarisiz oldu.")
        return None

    json_yolu, txt_yolu = save_reports(girdi_dosyasi, data, llm.model_name)
    logger.info("🎉 Otomatik trim suggestion tamamlandi.")
    return {
        "data": data,
        "json_path": json_yolu,
        "txt_path": txt_yolu,
    }

