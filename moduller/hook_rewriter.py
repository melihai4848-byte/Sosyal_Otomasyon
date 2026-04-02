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

logger = get_logger("hook")
HOOK_IDEA_COUNT = 6
HOOK_DIRECT_MODE_MAX_CHARS = 1600


def prepare_intro_transcript(girdi_dosyasi: Path, blok_limiti: int = 12, max_karakter: int = 5000) -> str:
    icerik = read_srt_file(girdi_dosyasi)
    bloklar = parse_srt_blocks(icerik)
    satirlar = [f"[{b.timing_line}] {b.text_content}" for b in bloklar if b.is_processable][:blok_limiti]
    return "\n".join(satirlar)[:max_karakter]


def build_critic_summary(critic_data: Optional[dict]) -> str:
    if not isinstance(critic_data, dict):
        return "AI Critic verisi yok. Yalnizca transcript acilisina gore karar ver."

    sorunlar = critic_data.get("biggest_issues", [])
    retention = critic_data.get("retention_analysis", {})

    return (
        f"Ozet: {critic_data.get('summary', '')}\n"
        f"Hook skoru: {critic_data.get('score_breakdown', {}).get('hook_strength', 0)}\n"
        f"Ilk 30 saniye analizi: {retention.get('first_30_seconds', '')}\n"
        f"Buyuk sorunlar: {' | '.join(sorunlar)}"
    ).strip()


def build_prompt(acilis_transkripti: str, critic_ozeti: str, feedback_ozeti: str) -> str:
    return f"""
Sen ust duzey bir YouTube hook strategist, retention copywriter ve opening editor'sun.

Gorevin:
Asagidaki video acilisini daha yuksek dikkat cekme ve izleyici tutma potansiyeliyle yeniden yaz.

KRITIK KURALLAR:
- Once mevcut acilisin en zayif kisimlarini ve izleyicinin dikkatini neyin daha hizli cekecegini kisaca analiz edebilirsin.
- Analiz kismini yalnizca duz metin veya <brainstorm> etiketiyle ver; {{ }}, [ ] veya kod blogu kullanma.
- Analizden sonra cevabini tek bir gecerli JSON nesnesi ya da tek bir ```json``` blogu olarak bitir.
- Tum ana metinler Turkce olsun.
- Kanca daha sert, net ve merak uyandirici olsun.
- Acilis vaadi, risk, surpriz veya odul mekanigi icersin.
- Asiri clickbait yapma; videonun gercek vaadine sadik kal.
- En az 3 alternatif hook uret.

JSON SEMASI:
{{
  "current_hook_problem": "",
  "recommended_primary_hook": "",
  "improved_hooks": [
    {{
      "hook": "",
      "why_it_works": "",
      "best_use_case": ""
    }}
  ],
  "opening_flow": [
    {{
      "segment": "0-5 sn",
      "goal": "",
      "script": ""
    }}
  ],
  "cta_bridge": "",
  "editor_notes": ["", "", ""]
}}

AI Critic Ozeti:
{critic_ozeti}

Retention Geri Bildirim Ozeti:
{feedback_ozeti}

Mevcut acilis transcripti:
{acilis_transkripti}
""".strip()


def build_ideation_prompt(acilis_transkripti: str, critic_ozeti: str, feedback_ozeti: str) -> str:
    return f"""
Sen ust duzey bir YouTube hook strategist'i, retention copywriter'i ve acilis edit planlayicisisin.

Gorevin:
Asagidaki acilis transcripti icin {HOOK_IDEA_COUNT} farkli hook yonu uret.

KRITIK KURALLAR:
- Final JSON'dan once kisa bir <brainstorm> bolumunde acilisin zayif noktalarini ve hangi hook psikolojilerinin daha iyi calisacagini dusun.
- <brainstorm> bolumunde yalnizca duz metin kullan; {{ }}, [ ] veya kod blogu kullanma.
- Final cevabinin sonunda tek bir gecerli JSON nesnesi ya da tek bir ```json``` blogu ver.
- Tum metin alanlari Turkce olsun.
- Her hook adayi farkli bir psikolojik aci kullansin: risk, odul, beklenmeyen sonuc, gizli maliyet, sert karsitlik, itiraz kirma gibi.
- Videonun gercek vaadini bozma.

JSON SEMASI:
{{
  "current_hook_problem": "",
  "candidate_hooks": [
    {{
      "hook": "",
      "angle": "",
      "why_it_works": "",
      "best_use_case": ""
    }}
  ],
  "editor_notes": ["", "", ""]
}}

AI Critic Ozeti:
{critic_ozeti}

Retention Geri Bildirim Ozeti:
{feedback_ozeti}

Mevcut acilis transcripti:
{acilis_transkripti}
""".strip()


def build_selection_prompt(acilis_transkripti: str, critic_ozeti: str, feedback_ozeti: str, ideation_payload: dict) -> str:
    ideation_text = json.dumps(ideation_payload, ensure_ascii=False, indent=2)
    return f"""
Sen ust duzey bir YouTube hook strategist'i, retention copywriter'i ve opening editorusun.

Gorevin:
Verilen hook adaylarini incele, gerekirse ufak cilalar yap ve final acilis yapisini cikar.

KRITIK KURALLAR:
- Once aday hook'lar icindeki guclu ve zayif yonleri, hangilerinin retention'a daha fazla hizmet ettigini kisaca analiz edebilirsin.
- Analiz kismini yalnizca duz metin veya <brainstorm> etiketiyle ver; {{ }}, [ ] veya kod blogu kullanma.
- Analizden sonra cevabini tek bir gecerli JSON nesnesi ya da tek bir ```json``` blogu olarak bitir.
- Tum ana metinler Turkce olsun.
- En az 3 alternatif hook uret.
- "opening_flow" gercekten editorun kullanabilecegi kadar somut olsun.
- Asiri clickbait yapma; videonun gercek vaadine sadik kal.

JSON SEMASI:
{{
  "current_hook_problem": "",
  "recommended_primary_hook": "",
  "improved_hooks": [
    {{
      "hook": "",
      "why_it_works": "",
      "best_use_case": ""
    }}
  ],
  "opening_flow": [
    {{
      "segment": "0-5 sn",
      "goal": "",
      "script": ""
    }}
  ],
  "cta_bridge": "",
  "editor_notes": ["", "", ""]
}}

AI Critic Ozeti:
{critic_ozeti}

Retention Geri Bildirim Ozeti:
{feedback_ozeti}

Hook Aday Havuzu:
{ideation_text}

Mevcut acilis transcripti:
{acilis_transkripti}
""".strip()


def normalize_data(data: dict) -> dict:
    if not isinstance(data, dict):
        return {}

    data.setdefault("current_hook_problem", "")
    data.setdefault("recommended_primary_hook", "")
    data.setdefault("improved_hooks", [])
    data.setdefault("opening_flow", [])
    data.setdefault("cta_bridge", "")
    data.setdefault("editor_notes", [])
    data.setdefault("skipped_by_routing", False)
    data.setdefault("routing_reason", "")

    data["current_hook_problem"] = normalize_whitespace(data.get("current_hook_problem", ""))
    data["recommended_primary_hook"] = normalize_whitespace(data.get("recommended_primary_hook", ""))
    data["cta_bridge"] = normalize_whitespace(data.get("cta_bridge", ""))
    data["editor_notes"] = [normalize_whitespace(x) for x in data.get("editor_notes", []) if normalize_whitespace(x)]
    data["routing_reason"] = normalize_whitespace(data.get("routing_reason", ""))
    data["skipped_by_routing"] = bool(data.get("skipped_by_routing"))

    temiz_hooks = []
    for item in data.get("improved_hooks", []):
        if not isinstance(item, dict):
            continue
        hook = normalize_whitespace(item.get("hook", ""))
        why = normalize_whitespace(item.get("why_it_works", ""))
        use_case = normalize_whitespace(item.get("best_use_case", ""))
        if hook:
            temiz_hooks.append({
                "hook": hook,
                "why_it_works": why,
                "best_use_case": use_case,
            })
    data["improved_hooks"] = temiz_hooks

    temiz_flow = []
    for item in data.get("opening_flow", []):
        if not isinstance(item, dict):
            continue
        segment = normalize_whitespace(item.get("segment", ""))
        goal = normalize_whitespace(item.get("goal", ""))
        script = normalize_whitespace(item.get("script", ""))
        if segment or script:
            temiz_flow.append({
                "segment": segment,
                "goal": goal,
                "script": script,
            })
    data["opening_flow"] = temiz_flow
    return data


def _routing_decision(critic_data: Optional[dict]) -> Tuple[Optional[bool], str]:
    if not isinstance(critic_data, dict):
        return None, ""

    routing = critic_data.get("routing_decisions", {})
    if not isinstance(routing, dict):
        return None, ""

    item = routing.get("hook_rewrite", {})
    if not isinstance(item, dict):
        return None, ""

    if "run" not in item:
        return None, normalize_whitespace(item.get("reason", ""))

    return bool(item.get("run")), normalize_whitespace(item.get("reason", ""))


def _should_run_hook_ideation(acilis: str, critic_data: Optional[dict]) -> bool:
    text_len = len(normalize_whitespace(acilis))
    if text_len >= HOOK_DIRECT_MODE_MAX_CHARS:
        return True

    if not isinstance(critic_data, dict):
        return False

    breakdown = critic_data.get("score_breakdown", {})
    hook_strength = 0
    try:
        hook_strength = int(breakdown.get("hook_strength", 0) or 0)
    except Exception:
        hook_strength = 0

    if hook_strength <= 45:
        return True
    if len(critic_data.get("rewrite_opportunities", []) or []) >= 2:
        return True
    if len(critic_data.get("biggest_issues", []) or []) >= 4:
        return True
    return False


def _build_skipped_data(reason: str, critic_data: Optional[dict]) -> dict:
    note = normalize_whitespace(reason) or "AI Critic acilisin yeterince guclu oldugunu dusunuyor."
    hook_strength = ""
    if isinstance(critic_data, dict):
        hook_strength = str(critic_data.get("score_breakdown", {}).get("hook_strength", ""))

    return normalize_data(
        {
            "current_hook_problem": (
                "AI Critic ciddi bir hook problemi tespit etmedi."
                + (f" Mevcut hook gucu skoru: {hook_strength}/100." if hook_strength else "")
            ),
            "recommended_primary_hook": "",
            "improved_hooks": [],
            "opening_flow": [],
            "cta_bridge": "",
            "editor_notes": [
                note,
                "Tam hook rewrite yerine mevcut acilis korunabilir.",
                "Bu rapor maliyeti azaltmak icin AI Critic routing kararina gore hafif modda olusturuldu.",
            ],
            "skipped_by_routing": True,
            "routing_reason": note,
        }
    )


def build_report_text(girdi_stem: str, data: dict, model_adi: str) -> str:
    if data.get("skipped_by_routing"):
        lines = [
            f"=== {girdi_stem} ICIN HOOK REWRITE RAPORU ===",
            f"Kullanilan Model: {model_adi}",
            "",
            "ROUTING KARARI",
            "-" * 50,
            data.get("routing_reason", ""),
            "",
            "MEVCUT DURUM",
            "-" * 50,
            data.get("current_hook_problem", ""),
            "",
            "EDITOR NOTLARI",
            "-" * 50,
        ]
        for item in data.get("editor_notes", []):
            lines.append(f"- {item}")
        return "\n".join(lines).strip() + "\n"

    lines = [
        f"=== {girdi_stem} ICIN HOOK REWRITE RAPORU ===",
        f"Kullanilan Model: {model_adi}",
        "",
        "MEVCUT HOOK PROBLEMI",
        "-" * 50,
        data.get("current_hook_problem", ""),
        "",
        "ONERILEN ANA HOOK",
        "-" * 50,
        data.get("recommended_primary_hook", ""),
        "",
        "HOOK ALTERNATIFLERI",
        "-" * 50,
    ]

    for idx, item in enumerate(data.get("improved_hooks", []), 1):
        lines.append(f"Alternatif {idx}: {item.get('hook', '')}")
        lines.append(f"Neden ise yarar: {item.get('why_it_works', '')}")
        lines.append(f"Kullanim senaryosu: {item.get('best_use_case', '')}")
        lines.append("")

    lines.extend([
        "ACILIS AKISI",
        "-" * 50,
    ])
    for item in data.get("opening_flow", []):
        lines.append(f"Segment: {item.get('segment', '')}")
        lines.append(f"Amaç: {item.get('goal', '')}")
        lines.append(f"Script: {item.get('script', '')}")
        lines.append("")

    lines.extend([
        "CTA KOPRUSU",
        "-" * 50,
        data.get("cta_bridge", ""),
        "",
        "EDITOR NOTLARI",
        "-" * 50,
    ])
    for item in data.get("editor_notes", []):
        lines.append(f"- {item}")

    return "\n".join(lines).strip() + "\n"


def analyze(
    girdi_dosyasi: Path,
    llm: CentralLLM,
    critic_data: Optional[dict] = None,
    feedback_data: Optional[dict] = None,
    intro_transcript: Optional[str] = None,
    critic_summary: Optional[str] = None,
    feedback_summary: Optional[str] = None,
    respect_routing: bool = False,
) -> dict:
    acilis = normalize_whitespace(intro_transcript) or prepare_intro_transcript(girdi_dosyasi)
    if not normalize_whitespace(acilis):
        logger.error("Hook rewrite icin acilis transcripti bulunamadi.")
        return {}

    routing_run, routing_reason = _routing_decision(critic_data)
    if respect_routing and routing_run is False:
        logger.info("AI Critic hook rewrite adimini gereksiz gordu; hafif skip raporu olusturuluyor.")
        return _build_skipped_data(routing_reason, critic_data)

    feedback_data = feedback_data or load_latest_feedback_data()
    critic_ozeti = normalize_whitespace(critic_summary) or build_critic_summary(critic_data)
    feedback_ozeti = normalize_whitespace(feedback_summary) or build_feedback_summary(feedback_data)
    logger.info("Hook Rewriter acilisi yeniden paketliyor...")
    parsed = None
    if _should_run_hook_ideation(acilis, critic_data):
        ideation_prompt = build_ideation_prompt(acilis, critic_ozeti, feedback_ozeti)
        ideation = extract_json_response(
            call_with_youtube_profile(
                llm,
                ideation_prompt,
                profile="creative_ideation",
            ),
            logger_override=logger,
        )
        if isinstance(ideation, dict) and isinstance(ideation.get("candidate_hooks"), list):
            parsed = extract_json_response(
                call_with_youtube_profile(
                    llm,
                    build_selection_prompt(acilis, critic_ozeti, feedback_ozeti, ideation),
                    profile="creative_ranker",
                ),
                logger_override=logger,
            )
    if not parsed:
        parsed = extract_json_response(
            call_with_youtube_profile(
                llm,
                build_prompt(acilis, critic_ozeti, feedback_ozeti),
                profile="creative_ranker",
            ),
            logger_override=logger,
        )
    if not parsed:
        logger.error("Hook Rewriter cevabi parse edilemedi.")
        return {}
    return normalize_data(parsed)


def save_reports(girdi_dosyasi: Path, data: dict, model_adi: str, save_txt: bool = True) -> Tuple[Path, Optional[Path]]:
    json_yolu = stem_json_output_path(girdi_dosyasi.stem, "_hook_rewrite.json", group="youtube")
    txt_yolu = txt_output_path("hook_rewrite") if save_txt else None

    with open(json_yolu, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    if txt_yolu is not None:
        txt_yolu.write_text(build_report_text(girdi_dosyasi.stem, data, model_adi), encoding="utf-8")
    logger.info(f"Hook rewrite JSON kaydedildi: {json_yolu.name}")
    if txt_yolu is not None:
        logger.info(f"Hook rewrite TXT kaydedildi: {txt_yolu.name}")
    return json_yolu, txt_yolu


def run():
    print("\n" + "=" * 60)
    print("HOOK ÜRETİCİ | Açılış Kancası Güçlendirme")
    print("=" * 60)

    girdi = select_primary_srt(logger, "Hook Uretici")
    if not girdi:
        return

    saglayici, model_adi = select_llm("smart")
    llm = CentralLLM(provider=saglayici, model_name=model_adi)

    data = analyze(girdi, llm)
    if not data:
        return logger.error("❌ Hook rewrite uretilemedi.")

    save_reports(girdi, data, model_adi)
    logger.info("🎉 Hook Rewriter islemi tamamlandi.")


def run_automatic(
    girdi_dosyasi: Path,
    llm: CentralLLM,
    critic_data: Optional[dict] = None,
    feedback_data: Optional[dict] = None,
    intro_transcript: Optional[str] = None,
    critic_summary: Optional[str] = None,
    feedback_summary: Optional[str] = None,
    respect_routing: bool = False,
    save_txt: bool = True,
) -> Optional[dict]:
    logger.info(f"🔄 OTOMASYON: {girdi_dosyasi.name} icin hook rewrite uretiliyor...")
    data = analyze(
        girdi_dosyasi,
        llm,
        critic_data=critic_data,
        feedback_data=feedback_data,
        intro_transcript=intro_transcript,
        critic_summary=critic_summary,
        feedback_summary=feedback_summary,
        respect_routing=respect_routing,
    )
    if not data:
        logger.error("❌ Otomatik hook rewrite basarisiz oldu.")
        return None

    json_yolu, txt_yolu = save_reports(girdi_dosyasi, data, llm.model_name, save_txt=save_txt)
    logger.info("🎉 Otomatik hook rewrite tamamlandi.")
    return {
        "data": data,
        "json_path": json_yolu,
        "txt_path": txt_yolu,
    }

