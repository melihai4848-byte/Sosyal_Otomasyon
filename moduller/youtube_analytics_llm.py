import json
import re
from typing import Optional

from moduller.llm_manager import CentralLLM, get_default_llm_config
from moduller.logger import get_logger
from moduller.youtube_llm_profiles import call_with_youtube_profile

logger = get_logger("YouTubeAnalyticsLLM")


def _analytics_depth_hint(mode: str) -> str:
    normalized = str(mode or "fast").strip().lower()
    if normalized == "deep":
        return "Daha kapsamli, biraz daha ayrintili ve stratejik derinligi yuksek yorumlar uret."
    return "Mumkun oldugunca net, kisa ve yuksek sinyal yogunlugunda kal. Gereksiz tekrar yapma."


def extract_json_response(llm_response: str) -> Optional[dict]:
    if not llm_response:
        return None
    try:
        response = re.sub(r"```json\s*|```", "", str(llm_response)).strip()
        try:
            parsed = json.loads(response)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

        start_positions = [index for index, char in enumerate(response) if char == "{"]
        for start in start_positions:
            depth = 0
            in_string = False
            escape = False
            for idx in range(start, len(response)):
                char = response[idx]
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
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        parsed = json.loads(response[start : idx + 1])
                        return parsed if isinstance(parsed, dict) else None
    except Exception as exc:
        logger.warning(f"Analytics LLM JSON parse hatasi: {exc}")
    return None


def build_analytics_llm() -> tuple[Optional[CentralLLM], dict]:
    try:
        provider, model_name = get_default_llm_config("smart")
        return (
            CentralLLM(provider=provider, model_name=model_name),
            {"enabled": True, "provider": provider, "model_name": model_name},
        )
    except Exception as exc:
        logger.warning(f"Analytics LLM hazirlanamadi: {exc}")
        return None, {"enabled": False, "provider": "", "model_name": "", "error": str(exc)}


def call_analytics_llm_json(
    llm: Optional[CentralLLM],
    prompt: str,
    profile: str = "analytic_json",
    retries: int = 2,
) -> Optional[dict]:
    if llm is None:
        return None
    last_error: Optional[Exception] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            response = call_with_youtube_profile(llm, prompt, profile=profile)
            parsed = extract_json_response(response)
            if parsed is not None:
                return parsed
        except Exception as exc:
            last_error = exc
            logger.warning(f"Analytics LLM cagri hatasi ({attempt}/{retries}): {exc}")
    if last_error:
        logger.warning(f"Analytics LLM fallback devreye girdi: {last_error}")
    return None


def build_channel_analysis_prompt(summary_data: dict) -> str:
    scoped_note = ""
    if isinstance(summary_data, dict) and str(summary_data.get("analysis_scope_label") or "").strip():
        scoped_note = (
            f"Bu veri kanal geneli degil, yalnizca {summary_data.get('analysis_scope_label')} formatinin alt kumesine aittir. "
            "Yorumunu sadece bu formatin performansi icin yap.\n"
        )
    return (
        "Sen deneyimli bir YouTube Analytics stratejistisin.\n"
        "Elindeki kanal verisini detayli ama yalnizca verilen metriklere dayanarak yorumla.\n"
        "Verilerde olmayan hicbir metriği uydurma. Tum ciktini Turkce yaz.\n"
        f"{scoped_note}"
        "JSON disinda hicbir sey donme.\n\n"
        "Asagidaki JSON seklinde cevap ver:\n"
        "{\n"
        '  "executive_summary": "tek paragraf",\n'
        '  "channel_diagnosis": ["4-6 madde"],\n'
        '  "what_is_working": ["4-6 madde"],\n'
        '  "what_is_not_working": ["4-6 madde"],\n'
        '  "content_pillars_to_push": ["3-5 madde"],\n'
        '  "content_pillars_to_reduce": ["3-5 madde"],\n'
        '  "title_packaging_notes": ["3-5 madde"],\n'
        '  "opportunities": ["3-5 madde"],\n'
        '  "risks": ["3-5 madde"]\n'
        "}\n\n"
        f"Kanal verisi:\n{json.dumps(summary_data, ensure_ascii=False, indent=2)}"
    )


def build_channel_action_prompt(summary_data: dict, llm_analysis: dict) -> str:
    scoped_note = ""
    if isinstance(summary_data, dict) and str(summary_data.get("analysis_scope_label") or "").strip():
        scoped_note = (
            f"Bu veri yalnizca {summary_data.get('analysis_scope_label')} formatina aittir. "
            "Aksiyon planini genel kanal yerine bu format icin spesifiklestir.\n"
        )
    return (
        "Sen sert ama yapici bir YouTube buyume danismanisin.\n"
        "Asagidaki kanal analizi ve ham veriye bakarak gercekci bir kritik ve 30 gunluk aksiyon plani hazirla.\n"
        f"{scoped_note}"
        "Tum ciktini Turkce yaz, JSON disinda bir sey donme.\n\n"
        "Asagidaki JSON seklinde cevap ver:\n"
        "{\n"
        '  "critical_verdict": "tek paragraf",\n'
        '  "must_fix_now": ["4-6 madde"],\n'
        '  "next_30_day_roadmap": ["7-10 madde"],\n'
        '  "high_priority_experiments": ["4-6 madde"],\n'
        '  "do_not_repeat": ["3-5 madde"]\n'
        "}\n\n"
        f"Kanal verisi:\n{json.dumps(summary_data, ensure_ascii=False, indent=2)}\n\n"
        f"Onceki yorum:\n{json.dumps(llm_analysis or {}, ensure_ascii=False, indent=2)}"
    )


def build_channel_combined_prompt(summary_data: dict, mode: str = "fast") -> str:
    scoped_note = ""
    if isinstance(summary_data, dict) and str(summary_data.get("analysis_scope_label") or "").strip():
        scoped_note = (
            f"Bu veri kanal geneli degil, yalnizca {summary_data.get('analysis_scope_label')} formatina aittir. "
            "Tespitlerini ve aksiyon planini bu formata gore yaz.\n"
        )
    return (
        "Sen deneyimli bir YouTube Analytics stratejisti ve buyume danismanisin.\n"
        "Elindeki kanal verisini yalnizca verilen metriklere dayanarak yorumla. Veride olmayan hicbir metrik uydurma.\n"
        "Tum ciktini Turkce yaz, JSON disinda hicbir sey donme.\n"
        f"{scoped_note}"
        f"{_analytics_depth_hint(mode)}\n\n"
        "Asagidaki JSON seklinde cevap ver:\n"
        "{\n"
        '  "llm_analysis": {\n'
        '    "executive_summary": "tek paragraf",\n'
        '    "channel_diagnosis": ["4-6 madde"],\n'
        '    "what_is_working": ["4-6 madde"],\n'
        '    "what_is_not_working": ["4-6 madde"],\n'
        '    "content_pillars_to_push": ["3-5 madde"],\n'
        '    "content_pillars_to_reduce": ["3-5 madde"],\n'
        '    "title_packaging_notes": ["3-5 madde"],\n'
        '    "opportunities": ["3-5 madde"],\n'
        '    "risks": ["3-5 madde"]\n'
        "  },\n"
        '  "critic_and_action_plan": {\n'
        '    "critical_verdict": "tek paragraf",\n'
        '    "must_fix_now": ["4-6 madde"],\n'
        '    "next_30_day_roadmap": ["7-10 madde"],\n'
        '    "high_priority_experiments": ["4-6 madde"],\n'
        '    "do_not_repeat": ["3-5 madde"]\n'
        "  }\n"
        "}\n\n"
        f"Kanal verisi:\n{json.dumps(summary_data, ensure_ascii=False, indent=2)}"
    )


def build_video_analysis_prompt(video_data: dict) -> str:
    return (
        "Sen deneyimli bir YouTube postmortem analistisin.\n"
        "Secilen videonun performansini kanal benchmarklari ve retention verisi ile birlikte yorumla.\n"
        "Yorum yaparken yalnizca verilen veriye dayan. Veride olmayan metrikleri uydurma.\n"
        "Tum ciktini Turkce yaz, JSON disinda bir sey donme.\n\n"
        "Asagidaki JSON seklinde cevap ver:\n"
        "{\n"
        '  "executive_summary": "tek paragraf",\n'
        '  "performance_story": "tek paragraf",\n'
        '  "strengths": ["4-6 madde"],\n'
        '  "weaknesses": ["4-6 madde"],\n'
        '  "root_causes": ["3-5 madde"],\n'
        '  "retention_diagnosis": ["3-5 madde"],\n'
        '  "packaging_notes": ["3-5 madde"],\n'
        '  "editing_notes": ["3-5 madde"],\n'
        '  "next_video_lessons": ["4-6 madde"]\n'
        "}\n\n"
        f"Video verisi:\n{json.dumps(video_data, ensure_ascii=False, indent=2)}"
    )


def build_video_action_prompt(video_data: dict, llm_analysis: dict) -> str:
    return (
        "Sen sert ama yapici bir YouTube strateji editorusun.\n"
        "Secilen video icin net bir kritik, aksiyon plani ve sonraki 3 video icin yol haritasi cikar.\n"
        "Tum ciktini Turkce yaz, JSON disinda bir sey donme.\n\n"
        "Asagidaki JSON seklinde cevap ver:\n"
        "{\n"
        '  "critical_verdict": "tek paragraf",\n'
        '  "must_fix_now": ["4-6 madde"],\n'
        '  "next_video_action_plan": ["5-8 madde"],\n'
        '  "next_3_video_roadmap": ["3-5 madde"],\n'
        '  "experiments": ["3-5 madde"],\n'
        '  "do_not_repeat": ["3-5 madde"]\n'
        "}\n\n"
        f"Video verisi:\n{json.dumps(video_data, ensure_ascii=False, indent=2)}\n\n"
        f"Onceki yorum:\n{json.dumps(llm_analysis or {}, ensure_ascii=False, indent=2)}"
    )


def build_video_combined_prompt(video_data: dict, mode: str = "fast") -> str:
    return (
        "Sen deneyimli bir YouTube postmortem analisti ve strateji editorusun.\n"
        "Secilen videonun performansini kanal benchmarklari ve retention verisi ile birlikte yorumla.\n"
        "Yorum yaparken yalnizca verilen veriye dayan. Veride olmayan metrikleri uydurma.\n"
        "Tum ciktini Turkce yaz, JSON disinda bir sey donme.\n"
        f"{_analytics_depth_hint(mode)}\n\n"
        "Asagidaki JSON seklinde cevap ver:\n"
        "{\n"
        '  "llm_analysis": {\n'
        '    "executive_summary": "tek paragraf",\n'
        '    "performance_story": "tek paragraf",\n'
        '    "strengths": ["4-6 madde"],\n'
        '    "weaknesses": ["4-6 madde"],\n'
        '    "root_causes": ["3-5 madde"],\n'
        '    "retention_diagnosis": ["3-5 madde"],\n'
        '    "packaging_notes": ["3-5 madde"],\n'
        '    "editing_notes": ["3-5 madde"],\n'
        '    "next_video_lessons": ["4-6 madde"]\n'
        "  },\n"
        '  "critic_and_action_plan": {\n'
        '    "critical_verdict": "tek paragraf",\n'
        '    "must_fix_now": ["4-6 madde"],\n'
        '    "next_video_action_plan": ["5-8 madde"],\n'
        '    "next_3_video_roadmap": ["3-5 madde"],\n'
        '    "experiments": ["3-5 madde"],\n'
        '    "do_not_repeat": ["3-5 madde"]\n'
        "  }\n"
        "}\n\n"
        f"Video verisi:\n{json.dumps(video_data, ensure_ascii=False, indent=2)}"
    )


def channel_analysis_fallback(summary_data: dict) -> dict:
    window = summary_data.get("window_metrics", {}) or {}
    benchmarks = summary_data.get("recent_video_benchmarks", {}) or {}
    return {
        "executive_summary": (
            f"Kanal son {window.get('window_days', 28)} gunde "
            f"{int(window.get('views', 0))} goruntulenme ve {int(window.get('net_subscribers', 0))} net abone degisimi uretmis."
        ),
        "channel_diagnosis": [
            f"Son pencere ortalama izlenme yuzdesi %{window.get('average_view_percentage', 0)}.",
            f"Median video goruntulenmesi {int(benchmarks.get('median_views', 0))}.",
            f"En cok tekrar eden anahtarlar: {', '.join(summary_data.get('title_keywords', [])[:6]) or 'yok'}.",
        ],
        "what_is_working": [
            "Kanal net bir tema etrafinda icerik biriktiriyor.",
            "Izleyicinin sorununu anlatan baslik eksenleri var.",
        ],
        "what_is_not_working": [
            "Benzer baslik ve paketleme kaliplari videolari birbirine yaklastiriyor.",
            "Performans dagilimi video bazinda oynaklik gosteriyor.",
        ],
        "content_pillars_to_push": ["Kazanan konu ailelerinin alt sorulari.", "Karar ve kiyas formatlari."],
        "content_pillars_to_reduce": ["Birbirine fazla benzeyen tekrar videolari."],
        "title_packaging_notes": ["Ayni kelime grubuna cok yaslanma.", "Faydayi daha spesifiklestir."],
        "opportunities": ["Basarili konulari alt acilara bolmek.", "Daha sert ilk vaadi test etmek."],
        "risks": ["Kendi iyi videolarini kopyalamak.", "Yavas acilan videolar."],
    }


def channel_action_fallback(summary_data: dict, llm_analysis: dict) -> dict:
    return {
        "critical_verdict": "Kanalin konusu net ama ayni konu ailesinde daha keskin acilar ve daha farkli paketleme gerekiyor.",
        "must_fix_now": [
            "Benzer baslik kaliplarini azalt.",
            "Her videoda tek bir net fayda vaadi belirle.",
            "Kazanan konulari alt sorulara bol.",
        ],
        "next_30_day_roadmap": [
            "Son 3 iyi videonun ortak temasini cikar.",
            "Her temadan 2 yeni alt baslik turet.",
            "Girislerde ilk 10 saniye vaadini netlestir.",
        ],
        "high_priority_experiments": [
            "Daha spesifik rakam/sonuc odakli baslik testi.",
            "Karar/kiyas acisi testi.",
        ],
        "do_not_repeat": ["Basligi ufak farkla tekrar paketleme.", "Faydayi gec gosteren acilis."],
    }


def video_analysis_fallback(video_data: dict) -> dict:
    selected = video_data.get("selected_video", {}) or {}
    comparison = video_data.get("comparison", {}) or {}
    retention = video_data.get("retention_analysis", {}) or {}
    return {
        "executive_summary": (
            f"Secilen video {selected.get('title', '')} kanal medianina gore goruntulenmede %{comparison.get('views_vs_median_pct', 0)} fark gosteriyor."
        ),
        "performance_story": "Video performansi paketleme, retention ve izleyici niyet uyumunun toplam sonucu olarak okunmali.",
        "strengths": [
            f"Ortalama izlenme yuzdesi %{selected.get('average_view_percentage', 0)} seviyesinde.",
            f"Net abone etkisi {selected.get('net_subscribers', 0)}.",
        ],
        "weaknesses": [
            f"Goruntulenme median farki %{comparison.get('views_vs_median_pct', 0)}.",
            f"Ilk 30 saniye kaybi %{retention.get('first_30s_drop_pct', 0)}.",
        ],
        "root_causes": [
            "Baslik-vaat ile icerik akisi tam oturmamis olabilir.",
            "Retention kirilmalarinda tempo kaybi veya gec payoff ihtimali var.",
        ],
        "retention_diagnosis": retention.get("patterns_to_avoid", [])[:4] or ["Retention verisi kritik not uretecek kadar guclu degil."],
        "packaging_notes": ["Baslik ve thumbnail vaadini ilk 20 saniye icinde karsila."],
        "editing_notes": retention.get("trim_suggester_guidance", [])[:4] or ["Orta bolum tempo kontrolleri yap."],
        "next_video_lessons": retention.get("next_video_policy", [])[:4] or ["Bir sonraki videoda daha net acilis kullan."],
    }


def video_action_fallback(video_data: dict, llm_analysis: dict) -> dict:
    return {
        "critical_verdict": "Secilen video, ozellikle acilis ve paketleme uyumu tarafinda daha sert optimizasyon gerektiriyor.",
        "must_fix_now": [
            "Baslik vaadini ilk 15 saniyede teslim et.",
            "Retention kirilmasi olan bolumlerde tekrar veya dolgu varsa temizle.",
            "Thumbnail ve baslikta tek fayda vaadini netlestir.",
        ],
        "next_video_action_plan": [
            "Bir sonraki videoda ilk cumleyi daha sert sonuc odakli yaz.",
            "Ilk 30 saniye icin mini payoff ekle.",
            "Orta bolumde pattern interrupt kullan.",
        ],
        "next_3_video_roadmap": [
            "Bir karar/kiyas videosu yayinla.",
            "Bir pratik rehber videosu yayinla.",
            "Bir video daha sert hook ile ciksin.",
        ],
        "experiments": [
            "Daha sert hook testi.",
            "Thumbnail metnini sadeleştirme.",
            "Ilk 30 saniyeyi kisaltma testi.",
        ],
        "do_not_repeat": ["Gec payoff veren acilis.", "Ayni faydayi tekrar eden arka plan cumleleri."],
    }
