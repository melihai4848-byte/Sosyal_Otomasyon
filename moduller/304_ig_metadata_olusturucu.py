import json
from pathlib import Path
from typing import Optional, Tuple

from moduller._module_alias import load_numbered_module
from moduller.logger import get_logger
from moduller.output_paths import stem_json_output_path, txt_output_path
from moduller.social_media_utils import load_related_json, select_primary_srt

_REELS_MODULE = load_numbered_module("302_reel_olusturucu.py")
load_latest_reels_data = _REELS_MODULE.load_latest_reels_data

_VIDEO_CRITIC_MODULE = load_numbered_module("202_video_critic.py")
normalize_whitespace = _VIDEO_CRITIC_MODULE.normalize_whitespace

logger = get_logger("ig_schedule")


def _collect_ranked_items(girdi_dosyasi: Path) -> dict:
    carousel_data = load_related_json(girdi_dosyasi, "_instagram_carousel.json")
    reels_data = load_latest_reels_data()
    story_data = load_related_json(girdi_dosyasi, "_instagram_story_plani.json")

    carousel_items = []
    if isinstance(carousel_data, dict):
        for item in carousel_data.get("carousel_candidates", []):
            if not isinstance(item, dict):
                continue
            carousel_items.append(
                {
                    "type": "Carousel",
                    "rank": int(item.get("rank", len(carousel_items) + 1) or len(carousel_items) + 1),
                    "score": int(item.get("viral_score", 0) or 0),
                    "title": normalize_whitespace(item.get("carousel_title_tr", "")),
                }
            )

    reel_items = []
    if isinstance(reels_data, dict):
        for item in reels_data.get("ideas", reels_data.get("reel_candidates", [])):
            if not isinstance(item, dict):
                continue
            reel_items.append(
                {
                    "type": "Reels",
                    "rank": int(item.get("rank", len(reel_items) + 1) or len(reel_items) + 1),
                    "score": int(item.get("viral_score", 0) or 0),
                    "title": normalize_whitespace(item.get("reel_title_tr", item.get("concept", ""))),
                }
            )

    story_items = []
    if isinstance(story_data, dict):
        for item in story_data.get("story_candidates", []):
            if not isinstance(item, dict):
                continue
            story_items.append(
                {
                    "type": "Story",
                    "rank": int(item.get("rank", len(story_items) + 1) or len(story_items) + 1),
                    "score": int(item.get("engagement_score", 0) or 0),
                    "title": normalize_whitespace(item.get("story_title_tr", "")),
                }
            )

    return {
        "carousel_items": sorted(carousel_items, key=lambda item: (-item["score"], item["rank"])),
        "reel_items": sorted(reel_items, key=lambda item: (-item["score"], item["rank"])),
        "story_items": sorted(story_items, key=lambda item: (-item["score"], item["rank"])),
    }


def _build_week_plan(all_items: list[dict], day_count: int = 7) -> list[dict]:
    gunler = [
        {
            "day": day,
            "items": [],
            "item_count": 0,
            "score_total": 0,
            "last_type": "",
        }
        for day in range(1, day_count + 1)
    ]

    for item in all_items:
        aday_gunler = sorted(
            gunler,
            key=lambda gun: (
                gun["item_count"],
                0 if gun["last_type"] != item["type"] else 1,
                gun["score_total"],
                gun["day"],
            ),
        )
        secilen_gun = aday_gunler[0]
        secilen_gun["items"].append(
            {
                "type": item["type"],
                "rank": item["rank"],
                "score": item["score"],
                "title": item.get("title", ""),
            }
        )
        secilen_gun["item_count"] += 1
        secilen_gun["score_total"] += item.get("score", 0)
        secilen_gun["last_type"] = item["type"]

    return [
        {
            "day": gun["day"],
            "items": gun["items"],
            "item_count": gun["item_count"],
        }
        for gun in gunler
    ]


def analyze(girdi_dosyasi: Path, llm=None, **_) -> dict:
    pools = _collect_ranked_items(girdi_dosyasi)
    toplam_carousel = len(pools["carousel_items"])
    toplam_reels = len(pools["reel_items"])
    toplam_story = len(pools["story_items"])

    tum_icerikler = sorted(
        pools["carousel_items"] + pools["reel_items"] + pools["story_items"],
        key=lambda item: (-item["score"], item["type"], item["rank"]),
    )
    if not tum_icerikler:
        logger.error("Paylasim takvimi icin kullanilabilir Instagram ciktilari bulunamadi.")
        return {}

    hafta_plani = _build_week_plan(tum_icerikler)
    toplam_icerik = len(tum_icerikler)
    coklu_gun_sayisi = sum(1 for gun in hafta_plani if len(gun.get("items", [])) > 1)
    ozet = (
        f"Toplam {toplam_carousel} carousel, {toplam_reels} reels ve {toplam_story} story olmak uzere "
        f"{toplam_icerik} Instagram icerigi bulundu. Bu iceriklerin tamami 7 gunluk takvime dagitildi; "
        f"hicbir icerik disarida birakilmadi."
    )
    if coklu_gun_sayisi:
        ozet += f" Yogun gunlerde birden fazla paylasim onerildi: {coklu_gun_sayisi} gun."

    return {
        "counts": {
            "carousel": toplam_carousel,
            "reels": toplam_reels,
            "story": toplam_story,
            "total_items": toplam_icerik,
        },
        "summary_tr": ozet,
        "weekly_plan": hafta_plani,
    }


def build_report_text(girdi_stem: str, data: dict) -> str:
    counts = data.get("counts", {})
    lines = [
        f"=== {girdi_stem} ICIN INSTAGRAM PAYLASIM TAKVIMI ===",
        "",
        "GENEL OZET",
        "-" * 60,
        data.get("summary_tr", ""),
        "",
        "URETILEN ICERIK SAYILARI",
        "-" * 60,
        f"Carousel Sayisi: {counts.get('carousel', 0)}",
        f"Reels Sayisi: {counts.get('reels', 0)}",
        f"Story Sayisi: {counts.get('story', 0)}",
        f"Toplam Icerik Sayisi: {counts.get('total_items', 0)}",
        "",
        "1 HAFTALIK YAYIN TAKVIMI",
        "-" * 60,
    ]

    for day_plan in data.get("weekly_plan", []):
        lines.append(f"{day_plan.get('day')}. gun")
        lines.append("-" * 40)
        items = day_plan.get("items", [])
        if not items:
            lines.append("Bu gun icin yeni ana paylasim onerisi yok. Yorum cevaplama veya performans takibi yapilabilir.")
            lines.append("")
            continue

        for item in items:
            lines.append(f"{item.get('rank')} nolu {item.get('type')}")
            if item.get("title"):
                lines.append(f"Baslik / Konsept: {item.get('title')}")
            lines.append(f"Skor: {item.get('score', 0)}/100")
            lines.append("")

        lines.append("")

    return "\n".join(lines).strip() + "\n"


def save_reports(girdi_dosyasi: Path, data: dict) -> Tuple[Path, Path]:
    json_yolu = stem_json_output_path(girdi_dosyasi.stem, "_instagram_schedule_plan.json", group="instagram")
    txt_yolu = txt_output_path("instagram_metadata")

    with open(json_yolu, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)

    txt_yolu.write_text(build_report_text(girdi_dosyasi.stem, data), encoding="utf-8")
    logger.info(f"Instagram paylasim takvimi JSON kaydedildi: {json_yolu.name}")
    logger.info(f"Instagram paylasim takvimi TXT kaydedildi: {txt_yolu.name}")
    return json_yolu, txt_yolu


def run():
    print("\n" + "=" * 60)
    print("ETKILESIM PLANLAYICI")
    print("=" * 60)

    girdi = select_primary_srt(logger, "Etkilesim Planlayici")
    if not girdi:
        return

    data = analyze(girdi)
    if not data:
        return logger.error("❌ Instagram paylasim takvimi uretilemedi.")

    save_reports(girdi, data)
    logger.info("🎉 Instagram paylasim takvimi olusturma islemi tamamlandi.")


def run_automatic(
    girdi_dosyasi: Path,
    llm=None,
    youtube_metadata_data: Optional[dict] = None,
    trend_data: Optional[dict] = None,
    draft_llm=None,
) -> Optional[dict]:
    del youtube_metadata_data, trend_data, draft_llm, llm
    logger.info(f"🔄 OTOMASYON: {girdi_dosyasi.name} icin Instagram paylasim takvimi uretiliyor...")
    data = analyze(girdi_dosyasi)
    if not data:
        logger.error("❌ Otomatik Instagram paylasim takvimi uretimi basarisiz oldu.")
        return None

    json_yolu, txt_yolu = save_reports(girdi_dosyasi, data)
    return {
        "data": data,
        "json_path": json_yolu,
        "txt_path": txt_yolu,
    }
