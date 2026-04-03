from __future__ import annotations

from dataclasses import dataclass, replace

from moduller.llm_role_table import load_llm_role_table_entries


@dataclass(frozen=True)
class ModuleEntry:
    number: str
    key: str
    group: str
    title: str
    description: str
    module_path: str
    launch_message: str
    run_function: str = "run"
    menu_icon: str = ""
    menu_title: str = ""
    menu_order: int = 0
    manual_module_path: str = ""
    manual_launch_message: str = ""
    manual_run_function: str = ""
    pipeline_enabled: bool = False
    pipeline_report_key: str = ""
    pipeline_dependencies: tuple[str, ...] = ()
    requires_main_llm: bool = False
    requires_smart_llm: bool = False


PIPELINE_EXECUTION_ORDER: tuple[str, ...] = (
    "subtitle",
    "grammar",
    "translation",
    "description",
    "critic",
    "broll",
    "thumbnail_main",
    "music",
    "carousel",
    "reels",
    "story",
    "ig_metadata",
)
PIPELINE_ORDER_INDEX = {key: index for index, key in enumerate(PIPELINE_EXECUTION_ORDER, start=1)}


MENU_GROUPS: tuple[tuple[str, str], ...] = (
    ("subtitle", "📝 ALTYAZI GRUBU 📝"),
    ("youtube", "📺 YOUTUBE GRUBU (16:9 Yatay Videolar) 📺"),
    ("instagram", "📱 INSTAGRAM GRUBU (9:16 Dikey Videolar) 📱"),
    ("research", "🔎 ARAŞTIRMA GRUBU 🔎"),
    ("tools", "🧰 ARACLAR (TOOLS) GRUBU 🧰"),
)


_BASE_MODULE_REGISTRY: tuple[ModuleEntry, ...] = (
    ModuleEntry(
        number="101",
        key="subtitle",
        group="subtitle",
        title="Altyazı Oluşturucu (TR)",
        description="Videodan Türkçe altyazı çıkarır",
        module_path="moduller.subtitle_generator",
        launch_message="Altyazı Oluşturucu başlatılıyor...",
        menu_icon="🎙️",
        pipeline_enabled=True,
        pipeline_report_key="1_subtitle_generation",
    ),
    ModuleEntry(
        number="102",
        key="grammar",
        group="subtitle",
        title="Gramer Düzenleyici (TR)",
        description="Altyazı dilini düzeltir",
        module_path="moduller.subtitle_grammar_editor",
        launch_message="Gramer Düzenleyici başlatılıyor...",
        menu_icon="✨",
        pipeline_enabled=True,
        pipeline_report_key="2_grammar_fix",
        pipeline_dependencies=("subtitle",),
        requires_main_llm=True,
    ),
    ModuleEntry(
        number="103",
        key="translation",
        group="subtitle",
        title="Altyazı Çevirmeni (EN-DE)",
        description="Altyazıyı dillere çevirir",
        module_path="moduller.subtitle_translator",
        launch_message="Altyazı Çevirmeni başlatılıyor...",
        menu_icon="🌍",
        pipeline_enabled=True,
        pipeline_report_key="3_translation",
        pipeline_dependencies=("grammar",),
    ),
    ModuleEntry(
        number="201",
        key="description",
        group="youtube",
        title="Video Açıklaması (Description) Oluşturucu (TR-EN-DE)",
        description="3 dilde açıklama, kisimlar, hashtag ve baslik onerileri uretir",
        module_path="moduller.youtube_description_generator",
        launch_message="Video Açıklaması (Description) Oluşturucu (TR-EN-DE) başlatılıyor...",
        menu_title="Video Açıklama ve Başlık Oluşturucu (TR-EN-DE)",
        menu_order=4,
        menu_icon="📝",
        pipeline_enabled=True,
        pipeline_report_key="4_description",
        pipeline_dependencies=("grammar",),
        requires_smart_llm=True,
    ),
    ModuleEntry(
        number="202",
        key="critic",
        group="youtube",
        title="Video Eleştirmeni",
        description="Videoyu detayli analiz eder, Kesilecek kısımları belirler, Açılış cümleleri güçlendirir",
        module_path="moduller.video_critic",
        launch_message="Video Eleştirmeni başlatılıyor...",
        menu_title="Video Analiz Üretici",
        menu_order=5,
        manual_module_path="moduller.video_analysis_generator",
        manual_launch_message="Video Analiz Üretici başlatılıyor...",
        menu_icon="🎯",
        pipeline_enabled=True,
        pipeline_report_key="8_critic",
        pipeline_dependencies=("grammar",),
        requires_main_llm=True,
        requires_smart_llm=True,
    ),
    ModuleEntry(
        number="203",
        key="broll",
        group="youtube",
        title="B-Roll Prompt Üretici (16:9 Yatay)",
        description="Sahneye uygun B-roll promptları",
        module_path="moduller.broll_prompt_generator",
        launch_message="B-Roll Prompt Üretici başlatılıyor...",
        menu_icon="🎞️",
        menu_order=6,
        pipeline_enabled=True,
        pipeline_report_key="5_broll",
        pipeline_dependencies=("grammar",),
        requires_smart_llm=True,
    ),
    ModuleEntry(
        number="204",
        key="thumbnail_main",
        group="youtube",
        title="Thumbnail Prompt Üretici (16:9 Yatay)",
        description="Ana video görsel promptları",
        module_path="moduller.thumbnail_prompt_generator",
        launch_message="16:9 Thumbnail Prompt Üretici başlatılıyor...",
        run_function="run_youtube_thumbnail",
        menu_icon="🖼️",
        menu_order=7,
        pipeline_enabled=True,
        pipeline_report_key="9_thumbnail_main",
        pipeline_dependencies=("grammar",),
        requires_smart_llm=True,
    ),
    ModuleEntry(
        number="205",
        key="music",
        group="youtube",
        title="Müzik Prompt Oluşturucu",
        description="Background müzik planı ve İngilizce prompt üretir",
        module_path="moduller.music_prompt_generator",
        launch_message="Müzik Prompt Oluşturucu başlatılıyor...",
        menu_title="Müzik Prompt Üretici",
        manual_launch_message="Müzik Prompt Üretici başlatılıyor...",
        menu_icon="🎵",
        menu_order=8,
        pipeline_enabled=True,
        pipeline_report_key="16_music",
        pipeline_dependencies=("grammar",),
        requires_main_llm=True,
    ),
    ModuleEntry(
        number="301",
        key="carousel",
        group="instagram",
        title="Carousel Fikir Üretici",
        description="Carousel metin akışı hazırlar, uygun görsel promptlari üretir, uygun aciklamalar üretir.",
        module_path="moduller.instagram_carousel_generator",
        launch_message="Carousel Fikir Üretici başlatılıyor...",
        menu_icon="📚",
        menu_order=12,
        pipeline_enabled=True,
        pipeline_report_key="11_carousel",
        pipeline_dependencies=("grammar",),
        requires_main_llm=True,
        requires_smart_llm=True,
    ),
    ModuleEntry(
        number="302",
        key="reels",
        group="instagram",
        title="Reels Fikir Üretici",
        description="Uzun videodan anlamli Reels konseptleri çıkarır, uygun görsel promptları üretir",
        module_path="moduller.instagram_reels_generator",
        launch_message="Reels Fikir Üretici başlatılıyor...",
        menu_icon="🎬",
        menu_order=13,
        pipeline_enabled=True,
        pipeline_report_key="10_reels",
        pipeline_dependencies=("grammar",),
        requires_main_llm=True,
        requires_smart_llm=True,
    ),
    ModuleEntry(
        number="303",
        key="story",
        group="instagram",
        title="Story Serisi Fikir Üretici",
        description="Story akışı ve etkileşim planı konseptleri çıkarır",
        module_path="moduller.instagram_story_generator",
        launch_message="Story Serisi Fikir Üretici başlatılıyor...",
        menu_icon="📲",
        menu_order=14,
        pipeline_enabled=True,
        pipeline_report_key="13_story",
        pipeline_dependencies=("grammar",),
        requires_main_llm=True,
        requires_smart_llm=True,
    ),
    ModuleEntry(
        number="304",
        key="ig_metadata",
        group="instagram",
        title="Etkileşim Planlayıcı",
        description="Haftalık Instagram paylaşım planı çıkarır",
        module_path="moduller.instagram_engagement_planner",
        launch_message="Etkileşim Planlayıcı başlatılıyor...",
        menu_order=15,
        menu_icon="🗓️",
        pipeline_enabled=True,
        pipeline_report_key="12_instagram_metadata",
        pipeline_dependencies=("carousel", "reels", "story"),
    ),
    ModuleEntry(
        number="501",
        key="reels_render",
        group="tools",
        title="Reels Oluşturucu",
        description="Olusturulmus Reels promptlarini kullanarak ana videodan kısa videolar olusturur",
        module_path="moduller.reels_renderer",
        launch_message="Reels Oluşturucu baslatiliyor...",
        menu_icon="🎥",
        menu_order=1,
    ),
    ModuleEntry(
        number="401",
        key="topic",
        group="research",
        title="YouTube Trends Konu Fikirleri",
        description="Trend analizi yaparak yeni video fikirleri bulur",
        module_path="topic_selection_engine.main",
        launch_message="YouTube Trends Konu Fikirleri baslatiliyor...",
        run_function="run_from_menu",
        menu_icon="💡",
    ),
    ModuleEntry(
        number="402",
        key="feedback",
        group="research",
        title="YouTube Analytics Analizi",
        description="Kanali veya secilen videoyu detayli analiz eder",
        module_path="moduller.youtube_analytics_feedback",
        launch_message="YouTube Analytics Analizi baslatiliyor...",
        menu_icon="📉",
        requires_main_llm=True,
        requires_smart_llm=True,
    ),
    ModuleEntry(
        number="502",
        key="youtube_uploader",
        group="tools",
        title="YouTube Draft Upload Engine",
        description="Mevcut private YouTube taslagini baslik, aciklama ve altyazilarla gunceller",
        module_path="moduller.youtube_draft_uploader",
        launch_message="YouTube Draft Upload Engine başlatılıyor...",
        menu_icon="☁️",
        menu_order=2,
    ),
    ModuleEntry(
        number="503",
        key="automatic_broll_downloader",
        group="tools",
        title="Automatic B-Roll Downloader",
        description="Pexels/Pixabay/Freepik/Coverr ile otomatik stok medya indirir",
        module_path="moduller.automatic_broll_downloader",
        launch_message="Automatic B-Roll Downloader baslatiliyor...",
        menu_icon="🎞️",
        menu_order=3,
    ),
    ModuleEntry(
        number="504",
        key="premiere_xml",
        group="tools",
        title="Premiere Pro XML Entegrasyonu",
        description="Trim ve B-Roll verisinden rough-cut XML uretir",
        module_path="moduller.premiere_xml_exporter",
        launch_message="Premiere Pro XML Entegrasyonu baslatiliyor...",
        menu_icon="🧩",
        menu_order=4,
    ),
    ModuleEntry(
        number="505",
        key="output_cleaner",
        group="tools",
        title="Cikti Temizleyici",
        description="Tum output klasorlerini, json cache'leri ve yerel log/state dosyalarini temizler",
        module_path="moduller.output_cleaner",
        launch_message="Cikti Temizleyici baslatiliyor...",
        menu_icon="🧹",
        menu_order=5,
    ),
)


def _apply_llm_role_table(registry: tuple[ModuleEntry, ...]) -> tuple[ModuleEntry, ...]:
    role_entries = load_llm_role_table_entries()
    patched_entries: list[ModuleEntry] = []
    for entry in registry:
        role_entry = role_entries.get(entry.number)
        if not role_entry:
            patched_entries.append(entry)
            continue
        patched_entries.append(
            replace(
                entry,
                requires_main_llm=role_entry.main_enabled,
                requires_smart_llm=role_entry.smart_enabled,
            )
        )
    return tuple(patched_entries)


MODULE_REGISTRY: tuple[ModuleEntry, ...] = _apply_llm_role_table(_BASE_MODULE_REGISTRY)


MODULE_BY_NUMBER = {entry.number: entry for entry in MODULE_REGISTRY}
MODULE_BY_KEY = {entry.key: entry for entry in MODULE_REGISTRY}
GROUP_TITLE_BY_KEY = dict(MENU_GROUPS)


def format_menu_label(entry: ModuleEntry) -> str:
    label = entry.menu_title or entry.title
    if entry.menu_icon:
        return f"{entry.menu_icon} {label} {entry.menu_icon}"
    return label


def _menu_sort_value(entry: ModuleEntry) -> int:
    return entry.menu_order or int(entry.number)


def iter_group_modules(group: str) -> list[ModuleEntry]:
    return sorted(
        [entry for entry in MODULE_REGISTRY if entry.group == group],
        key=_menu_sort_value,
    )


def iter_menu_sections() -> list[tuple[str, list[ModuleEntry]]]:
    grouped_entries: dict[str, list[ModuleEntry]] = {}
    for entry in MODULE_REGISTRY:
        grouped_entries.setdefault(entry.group, []).append(entry)

    sorted_sections = sorted(
        grouped_entries.items(),
        key=lambda item: min(int(entry.number) for entry in item[1]),
    )
    return [
        (
            GROUP_TITLE_BY_KEY.get(group_key, group_key),
            sorted(entries, key=_menu_sort_value),
        )
        for group_key, entries in sorted_sections
    ]


def get_module_by_number(number: str) -> ModuleEntry | None:
    return MODULE_BY_NUMBER.get(str(number))


def get_pipeline_modules() -> list[ModuleEntry]:
    return sorted(
        [
            entry
            for entry in MODULE_REGISTRY
            if entry.pipeline_enabled and entry.group not in {"research", "tools"}
        ],
        key=lambda entry: (PIPELINE_ORDER_INDEX.get(entry.key, 999), int(entry.number)),
    )
