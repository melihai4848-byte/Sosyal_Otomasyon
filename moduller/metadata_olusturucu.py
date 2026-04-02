import json
from pathlib import Path
from typing import Optional, Tuple

from moduller._module_alias import load_numbered_module
from moduller.llm_manager import CentralLLM, select_llm
from moduller.logger import get_logger
from moduller.output_paths import stem_json_output_path
from moduller.social_media_utils import select_primary_srt

_YOUTUBE_METADATA_MODULE = load_numbered_module("201_youtube_metadata_olusturucu.py")
DESCRIPTION_LANGUAGES = _YOUTUBE_METADATA_MODULE.ALL_LANGUAGES
description_otomatik_islem = _YOUTUBE_METADATA_MODULE.run_automatic

logger = get_logger("Metadata Wrapper")


def _select_srt() -> Optional[Path]:
    return select_primary_srt(logger, "Metadata Paketleyici")


def _select_languages() -> list[str]:
    print("\n" + "-" * 50)
    print("🌍 METADATA PAKETI DIL SECIMI")
    print("-" * 50)
    print("[1] Turkce")
    print("[2] Ingilizce")
    print("[3] Almanca")
    print("[4] Hepsi")

    secim = input("\n👉 Dil secimi: ").strip()
    mapping = {
        "1": ["Türkçe"],
        "2": ["İngilizce"],
        "3": ["Almanca"],
        "4": DESCRIPTION_LANGUAGES,
    }
    return mapping.get(secim, DESCRIPTION_LANGUAGES)


def _merge_language_data(description_data: dict, title_data: dict) -> dict:
    description_text = description_data.get("description_text", "")
    hashtag_line = description_data.get("hashtag_line", "")
    title_items = title_data.get("titles", [])
    hook_text = "\n".join(description_data.get("hook_lines", []))

    return {
        "titles": [item.get("title", "") for item in title_items if item.get("title")],
        "best_title": title_data.get("best_title", ""),
        "title_scoring": title_items,
        "description": {
            "hook": hook_text,
            "seo": hashtag_line,
        },
        "description_with_hashtags": description_data.get("description_with_hashtags", ""),
        "hashtags": description_data.get("hashtags", []),
        "tags": description_data.get("hashtags", []),
        "chapters": description_data.get("chapters", []),
        "search_terms": description_data.get("search_terms", []),
    }


def merge_metadata(description_payload: dict, title_payload: dict) -> dict:
    tum_sonuclar = {}
    diller = set(description_payload.keys()) | set(title_payload.keys())
    for dil in DESCRIPTION_LANGUAGES:
        if dil not in diller:
            continue
        desc = description_payload.get(dil, {})
        title = title_payload.get(dil, {})
        if not desc and not title:
            continue
        tum_sonuclar[dil] = _merge_language_data(desc, title)
    return tum_sonuclar


def build_report_text(girdi_stem: str, tum_sonuclar: dict, model_adi: str) -> str:
    satirlar = [
        f"=== {girdi_stem} ICIN BIRLESIK METADATA RAPORU ===",
        f"Kullanilan Model: {model_adi}",
        "Bu rapor yeni Description + Title modullerinden birlestirilerek olusturuldu.",
    ]

    for dil, data in tum_sonuclar.items():
        satirlar.extend(
            [
                "",
                "=" * 60,
                f"{dil.upper()} METADATA",
                "=" * 60,
                "",
                "BASLIKLAR",
                "-" * 40,
            ]
        )
        for item in data.get("title_scoring", []):
            satirlar.append(f"#{item.get('rank', '')} | {item.get('title', '')} | Skor: {item.get('score', 0)}")
            satirlar.append(f"Neden (TR): {item.get('reason_tr', '')}")

        satirlar.extend(
            [
                "",
                "DESCRIPTION",
                "-" * 40,
                data.get("description", {}).get("hook", ""),
                "",
                "KISIMLAR",
                "-" * 40,
                "\n".join(
                    f"{item.get('timestamp', '')} - {item.get('title', '')}"
                    for item in data.get("chapters", [])
                    if item.get("timestamp") and item.get("title")
                ),
                "",
                "HASHTAG SATIRI",
                "-" * 40,
                data.get("description", {}).get("seo", ""),
            ]
        )
    return "\n".join(satirlar).strip() + "\n"


def save_reports(girdi_dosyasi: Path, tum_sonuclar: dict, model_adi: str) -> Tuple[Path, Optional[Path]]:
    json_yolu = stem_json_output_path(girdi_dosyasi.stem, "_metadata.json", group="youtube")

    with open(json_yolu, "w", encoding="utf-8") as handle:
        json.dump(tum_sonuclar, handle, ensure_ascii=False, indent=2)

    logger.info(f"Birlesik metadata JSON kaydedildi: {json_yolu.name}")
    return json_yolu, None


def update_combined_metadata(girdi_dosyasi: Path, model_adi: str) -> Optional[Tuple[Path, Optional[Path]]]:
    description_json = stem_json_output_path(girdi_dosyasi.stem, "_video_description.json", group="youtube")
    title_json = stem_json_output_path(girdi_dosyasi.stem, "_video_titles.json", group="youtube")

    if not description_json.exists() or not title_json.exists():
        return None

    try:
        description_payload = json.loads(description_json.read_text(encoding="utf-8"))
        title_payload = json.loads(title_json.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Birlesik metadata senkronizasyonu basarisiz: {exc}")
        return None

    tum_sonuclar = merge_metadata(description_payload, title_payload)
    if not tum_sonuclar:
        return None
    return save_reports(girdi_dosyasi, tum_sonuclar, model_adi)


def run() -> None:
    print("\n" + "=" * 60)
    print("METADATA PAKETLEYICI | DESCRIPTION + TITLE")
    print("=" * 60)

    girdi = _select_srt()
    if not girdi:
        return

    hedef_diller = _select_languages()
    saglayici, model_adi = select_llm("smart")
    llm = CentralLLM(provider=saglayici, model_name=model_adi)

    sonuc = run_automatic(girdi, llm, llm, hedef_diller=hedef_diller)
    if not sonuc:
        logger.error("❌ Birlesik metadata paketleme basarisiz oldu.")
        return
    logger.info("🎉 Birlesik metadata paketleme tamamlandi.")


def run_automatic(
    girdi_dosyasi: Path,
    llm_step1: CentralLLM,
    llm_step2: Optional[CentralLLM] = None,
    trend_data: Optional[dict] = None,
    hedef_diller: Optional[list[str]] = None,
) -> Optional[dict]:
    logger.info(f"🔄 OTOMASYON: {girdi_dosyasi.name} icin birlesik metadata paketi olusturuluyor...")
    uretim_llm = llm_step2 or llm_step1
    hedef_diller = hedef_diller or DESCRIPTION_LANGUAGES

    description_result = description_otomatik_islem(
        girdi_dosyasi,
        uretim_llm,
        hedef_diller=hedef_diller,
        trend_data=trend_data,
    )

    if not description_result:
        logger.error("Description modulu sonuc veremedi.")
        return None

    combined_result = update_combined_metadata(girdi_dosyasi, uretim_llm.model_name)
    if not combined_result:
        logger.error("Birlesik metadata olusturulamadi.")
        return None

    json_yolu, txt_yolu = combined_result
    try:
        tum_sonuclar = json.loads(json_yolu.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error(f"Birlesik metadata JSON okunamadi: {exc}")
        return None

    return {
        "data": tum_sonuclar,
        "json_path": json_yolu,
        "txt_path": txt_yolu,
        "description_json_path": description_result["json_path"],
        "title_json_path": description_result.get("title_json_path"),
    }

