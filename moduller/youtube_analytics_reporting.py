import json
import re
from datetime import date, datetime
from statistics import median
from typing import Any, Optional


TURKISH_STOPWORDS = {
    "ve",
    "ile",
    "icin",
    "ama",
    "gibi",
    "daha",
    "neden",
    "nasil",
    "ne",
    "mi",
    "mu",
    "mü",
    "bir",
    "bu",
    "da",
    "de",
    "ki",
    "ya",
    "veya",
    "the",
    "and",
    "for",
    "you",
    "your",
    "olan",
    "gore",
    "kadar",
}

SHORTS_MAX_DURATION_SECONDS = 180


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value or "0").replace(",", ".")))
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value or "0").replace(",", "."))
    except Exception:
        return default


def safe_div(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return numerator / denominator


def rate(part: float, whole: float, scale: float = 100.0) -> float:
    return round(safe_div(part, whole) * scale, 2)


def format_number(value: Any) -> str:
    return f"{safe_int(value):,}".replace(",", ".")


def format_percent(value: Any) -> str:
    return f"%{safe_float(value):.2f}"


def format_duration_seconds(value: Any) -> str:
    total_seconds = max(0, safe_int(value))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def iso_to_date(value: str) -> Optional[date]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except Exception:
        return None


def days_ago(value: str) -> Optional[int]:
    parsed = iso_to_date(value)
    if not parsed:
        return None
    return max(0, (date.today() - parsed).days)


def extract_title_keywords(titles: list[str], limit: int = 10) -> list[str]:
    counts: dict[str, int] = {}
    for title in titles:
        for token in re.findall(r"[a-zA-Z0-9çğıöşüÇĞİÖŞÜ]+", str(title or "").lower()):
            if len(token) < 3 or token in TURKISH_STOPWORDS:
                continue
            counts[token] = counts.get(token, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ordered[:limit]]


def _median(values: list[float | int]) -> float:
    cleaned = [float(value) for value in values if value is not None]
    return float(median(cleaned)) if cleaned else 0.0


def video_views(video: dict) -> int:
    return safe_int(video.get("analytics_views") or video.get("view_count"))


def video_average_view_percentage(video: dict) -> float:
    return safe_float(video.get("analytics_average_view_percentage"))


def video_like_rate(video: dict) -> float:
    return rate(safe_int(video.get("like_count")), video_views(video))


def video_comment_rate(video: dict) -> float:
    return rate(safe_int(video.get("comment_count")), video_views(video))


def video_watch_hours(video: dict) -> float:
    return round(safe_int(video.get("analytics_estimated_minutes_watched")) / 60.0, 2)


def classify_video_format(video: dict) -> str:
    return "shorts" if safe_int(video.get("duration_seconds")) <= SHORTS_MAX_DURATION_SECONDS else "long_form"


def nearest_retention(points: list[dict], target_second: int) -> float:
    if not points:
        return 0.0
    nearest = min(points, key=lambda item: abs(item["second"] - target_second))
    return float(nearest["retention_pct"])


def dropoff_severity(delta_pct: float) -> str:
    if delta_pct <= -15:
        return "cok yuksek"
    if delta_pct <= -10:
        return "yuksek"
    if delta_pct <= -6:
        return "orta"
    return "dusuk"


def find_dropoff_points(points: list[dict]) -> list[dict]:
    dropoffs = []
    for index in range(1, len(points)):
        previous = points[index - 1]
        current = points[index]
        delta = current["retention_pct"] - previous["retention_pct"]
        if delta > -5:
            continue
        dropoffs.append(
            {
                "second": current["second"],
                "retention_pct": round(current["retention_pct"], 2),
                "delta_pct": round(delta, 2),
                "severity": dropoff_severity(delta),
                "note": (
                    "Acilis zayifligi ihtimali."
                    if current["second"] <= 15
                    else "Akis yavasliyor veya tekrar hissi artiyor."
                    if current["second"] <= 90
                    else "Gorsel destek veya konu gecisi gerekiyor olabilir."
                ),
            }
        )
    dropoffs.sort(key=lambda item: item["delta_pct"])
    return dropoffs[:8]


def build_retention_analysis(points: list[dict], context: Optional[dict] = None) -> dict:
    context = context or {}
    if not points:
        return {}

    start_retention = float(points[0]["retention_pct"])
    first_10 = round(max(0.0, start_retention - nearest_retention(points, 10)), 2)
    first_30 = round(max(0.0, start_retention - nearest_retention(points, 30)), 2)
    first_60 = round(max(0.0, start_retention - nearest_retention(points, 60)), 2)
    dropoffs = find_dropoff_points(points)

    patterns_to_avoid = []
    hook_guidance = []
    trim_guidance = []
    broll_guidance = []
    next_video_policy = []

    if first_10 >= 25:
        patterns_to_avoid.append("Ilk 10 saniyede sonuc vermeyen yavas girisler")
        hook_guidance.append("Ilk 3 saniyede risk, sonuc veya odulu net soyle.")
        hook_guidance.append("Selamlamayi ikinci cumleye it.")
        next_video_policy.append("Yeni videolarda ilk cumle dogrudan sorunu veya sonucu aciklayacak.")

    if first_30 >= 40:
        patterns_to_avoid.append("Uzun baglam kuran ve asil faydayi gec gosteren acilis")
        hook_guidance.append("10-15 saniye icinde neden kalinmasi gerektigini somutlastir.")
        trim_guidance.append("Ilk 30 saniyedeki tekrar eden baglam cumlelerini kes veya hizlandir.")

    for item in dropoffs[:5]:
        second = item["second"]
        if second <= 20:
            broll_guidance.append(f"{second}. saniye civarina pattern interrupt ekle.")
        elif second <= 90:
            trim_guidance.append(f"{second}. saniye civarindaki bolumde tekrar veya dolgu varsa kes.")
            broll_guidance.append(f"{second}. saniye civarinda gorsel gecis veya B-roll dusun.")
        else:
            trim_guidance.append(f"{second}. saniye sonrasinda tempo dusuyor; bolumu daha kisa tut.")

    if not trim_guidance:
        trim_guidance.append("Retention egrisi kritik orta bolum kirilimi gostermiyor; dolgu kisimlarini temizle.")
    if not broll_guidance:
        broll_guidance.append("Retention egrisinde sert orta bolum dususu yok; B-roll konu gecislerinde kullanilabilir.")
    if not hook_guidance:
        hook_guidance.append("Acilis gorece dengeli; daha net bir fayda cumlesi eklemek yeterli olabilir.")
    if not next_video_policy:
        next_video_policy.append("Bir sonraki videoda daha hizli acilis ve daha erken payoff kullan.")

    strongest_breaks = ", ".join([f"{item.get('second')} sn" for item in dropoffs[:3]]) or "yok"
    summary = (
        f"Ilk 10 saniyede %{first_10}, ilk 30 saniyede %{first_30}, ilk 60 saniyede %{first_60} izleyici kaybi goruldu. "
        f"En sert kirilmalar: {strongest_breaks}."
    )
    return {
        "video_title": str(context.get("video_title") or context.get("title") or "").strip(),
        "summary": summary,
        "first_10s_drop_pct": first_10,
        "first_30s_drop_pct": first_30,
        "first_60s_drop_pct": first_60,
        "dropoff_points": dropoffs,
        "patterns_to_avoid": list(dict.fromkeys(patterns_to_avoid)),
        "hook_rewriter_guidance": list(dict.fromkeys(hook_guidance)),
        "trim_suggester_guidance": list(dict.fromkeys(trim_guidance)),
        "broll_guidance": list(dict.fromkeys(broll_guidance)),
        "next_video_policy": list(dict.fromkeys(next_video_policy)),
    }


def build_feedback_summary(data: Optional[dict]) -> str:
    if not isinstance(data, dict):
        return "Gecmis analytics geri bildirimi yok."
    source = data.get("retention_analysis") if isinstance(data.get("retention_analysis"), dict) else data
    dropoffs = source.get("dropoff_points", []) or []
    key_points = " | ".join([f"{item.get('second')} sn" for item in dropoffs[:4]]) or "Yok"
    return (
        f"Ilk 10 saniye kaybi: %{source.get('first_10s_drop_pct', 0)}\n"
        f"Ilk 30 saniye kaybi: %{source.get('first_30s_drop_pct', 0)}\n"
        f"Kacinilacak kaliplar: {' | '.join(source.get('patterns_to_avoid', [])[:4]) or 'Yok'}\n"
        f"Onemli drop-off noktalar: {key_points}"
    ).strip()


def summarize_channel_snapshot(snapshot: dict) -> dict:
    channel = snapshot.get("channel", {}) or {}
    channel_summary = snapshot.get("channel_summary", {}) or {}
    recent_videos = list(snapshot.get("recent_videos", []) or [])
    channel_daily = list(snapshot.get("channel_daily", []) or [])

    uploads_last_28 = sum(1 for video in recent_videos if (days_ago(video.get("published_at", "")) or 9999) <= 28)
    median_views = _median([video_views(video) for video in recent_videos])
    median_apv = _median([video_average_view_percentage(video) for video in recent_videos])
    median_like_rate = _median([video_like_rate(video) for video in recent_videos])
    median_comment_rate = _median([video_comment_rate(video) for video in recent_videos])

    half = max(1, len(channel_daily) // 2)
    first_half_views = sum(safe_int(item.get("views")) for item in channel_daily[:half])
    second_half_views = sum(safe_int(item.get("views")) for item in channel_daily[half:])
    trend_pct = round((safe_div(second_half_views, first_half_views) - 1.0) * 100, 2) if first_half_views else 0.0

    summarized_videos = []
    for video in recent_videos:
        format_type = classify_video_format(video)
        summarized_videos.append(
            {
                "video_id": video.get("video_id", ""),
                "title": video.get("title", ""),
                "published_at": video.get("published_at", ""),
                "published_days_ago": days_ago(video.get("published_at", "")),
                "duration": video.get("duration_label", format_duration_seconds(video.get("duration_seconds"))),
                "duration_seconds": safe_int(video.get("duration_seconds")),
                "format_type": format_type,
                "views": video_views(video),
                "average_view_percentage": round(video_average_view_percentage(video), 2),
                "watch_hours": video_watch_hours(video),
                "like_rate_pct": video_like_rate(video),
                "comment_rate_pct": video_comment_rate(video),
                "net_subscribers": safe_int(video.get("analytics_net_subscribers")),
            }
        )

    def _summarize_format_bucket(format_key: str, label: str) -> dict:
        videos = [video for video in summarized_videos if video.get("format_type") == format_key]
        uploads_last_28 = sum(1 for video in videos if safe_int(video.get("published_days_ago"), 9999) <= 28)
        median_views_bucket = _median([safe_int(video.get("views")) for video in videos])
        median_apv_bucket = _median([safe_float(video.get("average_view_percentage")) for video in videos])
        median_like_bucket = _median([safe_float(video.get("like_rate_pct")) for video in videos])
        median_comment_bucket = _median([safe_float(video.get("comment_rate_pct")) for video in videos])
        average_views_bucket = round(
            safe_div(sum(safe_int(video.get("views")) for video in videos), len(videos)),
            2,
        ) if videos else 0.0
        average_apv_bucket = round(
            safe_div(sum(safe_float(video.get("average_view_percentage")) for video in videos), len(videos)),
            2,
        ) if videos else 0.0
        average_duration_bucket = round(
            safe_div(sum(safe_int(video.get("duration_seconds")) for video in videos), len(videos)),
            2,
        ) if videos else 0.0
        return {
            "format_type": format_key,
            "label": label,
            "video_count": len(videos),
            "uploads_last_28_days": uploads_last_28,
            "total_views": sum(safe_int(video.get("views")) for video in videos),
            "total_watch_hours": round(sum(safe_float(video.get("watch_hours")) for video in videos), 2),
            "average_views": average_views_bucket,
            "average_view_percentage": average_apv_bucket,
            "average_duration_seconds": average_duration_bucket,
            "median_views": round(median_views_bucket, 2),
            "median_average_view_percentage": round(median_apv_bucket, 2),
            "median_like_rate_pct": round(median_like_bucket, 2),
            "median_comment_rate_pct": round(median_comment_bucket, 2),
            "title_keywords": extract_title_keywords([video.get("title", "") for video in videos], limit=10),
            "top_videos": sorted(videos, key=lambda item: safe_int(item.get("views")), reverse=True)[:5],
        }

    shorts_summary = _summarize_format_bucket("shorts", "Shorts")
    long_form_summary = _summarize_format_bucket("long_form", "Uzun Video")

    return {
        "channel": {
            "channel_id": channel.get("channel_id", ""),
            "channel_title": channel.get("channel_title", ""),
            "description": channel.get("description", ""),
            "subscriber_count": safe_int(channel.get("subscriber_count")),
            "video_count": safe_int(channel.get("video_count")),
            "view_count": safe_int(channel.get("view_count")),
            "keywords": channel.get("keywords", ""),
        },
        "window_metrics": {
            "window_days": safe_int(channel_summary.get("window_days")),
            "views": safe_int(channel_summary.get("views")),
            "watch_hours": round(safe_int(channel_summary.get("estimated_minutes_watched")) / 60.0, 2),
            "average_view_duration_seconds": round(safe_float(channel_summary.get("average_view_duration_seconds")), 2),
            "average_view_percentage": round(safe_float(channel_summary.get("average_view_percentage")), 2),
            "net_subscribers": safe_int(channel_summary.get("net_subscribers")),
            "uploads_last_28_days": uploads_last_28,
            "view_trend_vs_previous_half_pct": trend_pct,
        },
        "recent_video_benchmarks": {
            "median_views": round(median_views, 2),
            "median_average_view_percentage": round(median_apv, 2),
            "median_like_rate_pct": round(median_like_rate, 2),
            "median_comment_rate_pct": round(median_comment_rate, 2),
        },
        "title_keywords": extract_title_keywords([video.get("title", "") for video in recent_videos], limit=12),
        "content_format_breakdown": {
            "shorts": shorts_summary,
            "long_form": long_form_summary,
        },
        "recent_videos": summarized_videos,
    }


def summarize_selected_video(snapshot: dict, selected_video: dict, retention_analysis: Optional[dict]) -> dict:
    peers = [video for video in (snapshot.get("recent_videos", []) or []) if video.get("video_id") != selected_video.get("video_id")]
    peers = peers or list(snapshot.get("recent_videos", []) or [])
    median_views = _median([video_views(video) for video in peers])
    median_apv = _median([video_average_view_percentage(video) for video in peers])
    median_like_rate = _median([video_like_rate(video) for video in peers])
    median_comment_rate = _median([video_comment_rate(video) for video in peers])
    median_net_subscribers = _median([safe_int(video.get("analytics_net_subscribers")) for video in peers])

    selected_views = video_views(selected_video)
    selected_apv = video_average_view_percentage(selected_video)
    selected_like = video_like_rate(selected_video)
    selected_comment = video_comment_rate(selected_video)
    selected_net_subscribers = safe_int(selected_video.get("analytics_net_subscribers"))

    return {
        "channel": {"channel_title": (snapshot.get("channel", {}) or {}).get("channel_title", "")},
        "selected_video": {
            "video_id": selected_video.get("video_id", ""),
            "title": selected_video.get("title", ""),
            "published_at": selected_video.get("published_at", ""),
            "published_days_ago": days_ago(selected_video.get("published_at", "")),
            "duration": selected_video.get("duration_label", format_duration_seconds(selected_video.get("duration_seconds"))),
            "views": selected_views,
            "average_view_percentage": round(selected_apv, 2),
            "watch_hours": video_watch_hours(selected_video),
            "like_rate_pct": round(selected_like, 2),
            "comment_rate_pct": round(selected_comment, 2),
            "net_subscribers": selected_net_subscribers,
        },
        "benchmark": {
            "median_views": round(median_views, 2),
            "median_average_view_percentage": round(median_apv, 2),
            "median_like_rate_pct": round(median_like_rate, 2),
            "median_comment_rate_pct": round(median_comment_rate, 2),
            "median_net_subscribers": round(median_net_subscribers, 2),
        },
        "comparison": {
            "views_vs_median_pct": round((safe_div(selected_views, median_views) - 1.0) * 100, 2) if median_views else 0.0,
            "average_view_percentage_delta": round(selected_apv - median_apv, 2),
            "like_rate_delta": round(selected_like - median_like_rate, 2),
            "comment_rate_delta": round(selected_comment - median_comment_rate, 2),
            "net_subscribers_delta": round(selected_net_subscribers - median_net_subscribers, 2),
        },
        "retention_analysis": retention_analysis or {},
    }


def build_channel_txt_report(payload: dict) -> str:
    summary = payload.get("channel_summary", {}) or {}
    llm_analysis = payload.get("llm_analysis", {}) or {}
    action_plan = payload.get("critic_and_action_plan", {}) or {}
    format_specific = payload.get("format_specific_analysis", {}) or {}
    recent_videos = summary.get("recent_videos", []) or []
    format_breakdown = summary.get("content_format_breakdown", {}) or {}
    shorts_summary = format_breakdown.get("shorts", {}) or {}
    long_form_summary = format_breakdown.get("long_form", {}) or {}
    lines = [
        "=== YOUTUBE ANALYTICS | KANAL ANALIZI ===",
        "",
        f"Kanal adi: {(summary.get('channel', {}) or {}).get('channel_title', '')}",
        f"Son pencere goruntulenme: {format_number(summary.get('window_metrics', {}).get('views'))}",
        f"Net abone: {format_number(summary.get('window_metrics', {}).get('net_subscribers'))}",
        f"Ortalama izlenme yuzdesi: {format_percent(summary.get('window_metrics', {}).get('average_view_percentage'))}",
        f"Son 28 gunde upload: {format_number(summary.get('window_metrics', {}).get('uploads_last_28_days'))}",
        f"Shorts sayisi: {format_number(shorts_summary.get('video_count'))} | Uzun video sayisi: {format_number(long_form_summary.get('video_count'))}",
        f"Anahtar kelimeler: {', '.join(summary.get('title_keywords', [])[:10]) or 'Yok'}",
        "",
        "YARATICI YAPAY ZEKA ANALIZI",
        "-" * 60,
        llm_analysis.get("executive_summary", "LLM yorumu uretilemedi."),
        "",
        "KANAL DIYAGNOZU",
        "-" * 60,
    ]
    for item in llm_analysis.get("channel_diagnosis", []):
        lines.append(f"- {item}")
    lines.extend(["", "NE CALISIYOR", "-" * 60])
    for item in llm_analysis.get("what_is_working", []):
        lines.append(f"- {item}")
    lines.extend(["", "NE CALISMIYOR", "-" * 60])
    for item in llm_analysis.get("what_is_not_working", []):
        lines.append(f"- {item}")
    lines.extend(["", "BUYUTULECEK ICERIK KOLLARI", "-" * 60])
    for item in llm_analysis.get("content_pillars_to_push", []):
        lines.append(f"- {item}")
    lines.extend(["", "AZALTILACAK ICERIK KOLLARI", "-" * 60])
    for item in llm_analysis.get("content_pillars_to_reduce", []):
        lines.append(f"- {item}")
    lines.extend(["", "PAKETLEME NOTLARI", "-" * 60])
    for item in llm_analysis.get("title_packaging_notes", []):
        lines.append(f"- {item}")
    lines.extend(["", "FIRSATLAR", "-" * 60])
    for item in llm_analysis.get("opportunities", []):
        lines.append(f"- {item}")
    lines.extend(["", "RISKLER", "-" * 60])
    for item in llm_analysis.get("risks", []):
        lines.append(f"- {item}")
    lines.extend(["", "FORMAT BAZLI OZET", "-" * 60])
    for bucket in (shorts_summary, long_form_summary):
        if not bucket or not safe_int(bucket.get("video_count")):
            continue
        lines.append(
            f"{bucket.get('label', '')}: {format_number(bucket.get('video_count'))} video | "
            f"Ort. goruntulenme {format_number(bucket.get('average_views'))} | "
            f"Ort. APV {format_percent(bucket.get('average_view_percentage'))}"
        )
        lines.append(
            f"Median goruntulenme {format_number(bucket.get('median_views'))} | "
            f"Son 28 gunde upload {format_number(bucket.get('uploads_last_28_days'))} | "
            f"Anahtarlar: {', '.join(bucket.get('title_keywords', [])[:6]) or 'Yok'}"
        )
        lines.append("")
    for format_key in ("shorts", "long_form"):
        block = format_specific.get(format_key, {}) or {}
        bucket = block.get("summary", {}) or {}
        scoped_analysis = block.get("llm_analysis", {}) or {}
        scoped_action = block.get("critic_and_action_plan", {}) or {}
        if not bucket or not safe_int(bucket.get("video_count")):
            continue
        lines.extend([f"{bucket.get('label', '').upper()} OZEL ANALIZI", "-" * 60])
        if scoped_analysis.get("executive_summary"):
            lines.append(scoped_analysis.get("executive_summary", ""))
        lines.extend(["", "NE CALISIYOR", "-" * 60])
        for item in scoped_analysis.get("what_is_working", []):
            lines.append(f"- {item}")
        lines.extend(["", "NE CALISMIYOR", "-" * 60])
        for item in scoped_analysis.get("what_is_not_working", []):
            lines.append(f"- {item}")
        lines.extend(["", "FORMAT ODAKLI FIRSATLAR", "-" * 60])
        for item in scoped_analysis.get("opportunities", []):
            lines.append(f"- {item}")
        lines.extend(["", "FORMAT ODAKLI RISKLER", "-" * 60])
        for item in scoped_analysis.get("risks", []):
            lines.append(f"- {item}")
        if scoped_action.get("critical_verdict"):
            lines.extend(["", "FORMAT KRITIGI", "-" * 60, scoped_action.get("critical_verdict", "")])
        must_fix_now = scoped_action.get("must_fix_now", []) or []
        if must_fix_now:
            lines.extend(["", "HEMEN DUZELT", "-" * 60])
            for item in must_fix_now:
                lines.append(f"- {item}")
        lines.append("")
    lines.extend(["", "SON 10 VIDEO OZETI", "-" * 60])
    for index, video in enumerate(recent_videos[:10], start=1):
        lines.append(
            f"{index}. {video.get('title', '')} | {format_number(video.get('views'))} izlenme | "
            f"{format_percent(video.get('average_view_percentage'))} APV"
        )
    lines.extend(["", "KRITIK", "-" * 60, action_plan.get("critical_verdict", "")])
    return "\n".join(lines).strip() + "\n"


def build_video_txt_report(payload: dict) -> str:
    video_summary = payload.get("video_summary", {}) or {}
    llm_analysis = payload.get("llm_analysis", {}) or {}
    action_plan = payload.get("critic_and_action_plan", {}) or {}
    retention = payload.get("retention_analysis", {}) or {}
    selected_video = (video_summary.get("selected_video", {}) or {})
    benchmark = video_summary.get("benchmark", {}) or {}
    comparison = video_summary.get("comparison", {}) or {}
    lines = [
        "=== YOUTUBE ANALYTICS | SECILEN VIDEO ANALIZI ===",
        "",
        f"Video: {selected_video.get('title', '')}",
        f"Goruntulenme: {format_number(selected_video.get('views'))}",
        f"Izlenme yuzdesi: {format_percent(selected_video.get('average_view_percentage'))}",
        f"Kanal median goruntulenme: {format_number(benchmark.get('median_views'))}",
        f"Goruntulenme farki: %{comparison.get('views_vs_median_pct', 0)}",
        f"APV delta: %{comparison.get('average_view_percentage_delta', 0)}",
        "",
        "RETENTION OZETI",
        "-" * 60,
        retention.get("summary", "Retention verisi alinamadi."),
        "",
        "YARATICI YAPAY ZEKA YORUMU",
        "-" * 60,
        llm_analysis.get("executive_summary", "LLM yorumu uretilemedi."),
        "",
        "PERFORMANS HIKAYESI",
        "-" * 60,
        llm_analysis.get("performance_story", ""),
        "",
        "GUCLU YANLAR",
        "-" * 60,
    ]
    for item in llm_analysis.get("strengths", []):
        lines.append(f"- {item}")
    lines.extend(["", "ZAYIF YANLAR", "-" * 60])
    for item in llm_analysis.get("weaknesses", []):
        lines.append(f"- {item}")
    lines.extend(["", "ROOT CAUSE", "-" * 60])
    for item in llm_analysis.get("root_causes", []):
        lines.append(f"- {item}")
    lines.extend(["", "RETENTION DIYAGNOZU", "-" * 60])
    for item in llm_analysis.get("retention_diagnosis", []):
        lines.append(f"- {item}")
    lines.extend(["", "PAKETLEME NOTLARI", "-" * 60])
    for item in llm_analysis.get("packaging_notes", []):
        lines.append(f"- {item}")
    lines.extend(["", "EDIT NOTLARI", "-" * 60])
    for item in llm_analysis.get("editing_notes", []):
        lines.append(f"- {item}")
    lines.extend(["", "SONRAKI VIDEO DERSLERI", "-" * 60])
    for item in llm_analysis.get("next_video_lessons", []):
        lines.append(f"- {item}")
    lines.extend([
        "",
        "KRITIK",
        "-" * 60,
        action_plan.get("critical_verdict", ""),
    ])
    return "\n".join(lines).strip() + "\n"


def build_action_plan_txt(payload: dict) -> str:
    action_plan = payload.get("critic_and_action_plan", {}) or {}
    experiment_items = (action_plan.get("high_priority_experiments", []) or []) + (action_plan.get("experiments", []) or [])
    lines = [
        "=== YOUTUBE ANALYTICS | KRITIK VE AKSIYON PLANI ===",
        "",
        action_plan.get("critical_verdict", ""),
        "",
        "HEMEN DUZELT",
        "-" * 60,
    ]
    for item in action_plan.get("must_fix_now", []):
        lines.append(f"- {item}")
    for section_title, key in (
        ("30 GUNLUK YOL HARITASI", "next_30_day_roadmap"),
        ("SONRAKI VIDEO AKSIYONLARI", "next_video_action_plan"),
        ("SONRAKI 3 VIDEO YOL HARITASI", "next_3_video_roadmap"),
        ("TEKRAR ETME", "do_not_repeat"),
    ):
        values = action_plan.get(key, []) or []
        if not values:
            continue
        lines.extend(["", section_title, "-" * 60])
        for item in values:
            lines.append(f"- {item}")
    if experiment_items:
        lines.extend(["", "TEST EDILECEK DENEYLER", "-" * 60])
        for item in experiment_items:
            lines.append(f"- {item}")
    return "\n".join(lines).strip() + "\n"


def build_channel_external_prompt(payload: dict) -> str:
    summary = payload.get("channel_summary", {}) or {}
    llm_analysis = payload.get("llm_analysis", {}) or {}
    action_plan = payload.get("critic_and_action_plan", {}) or {}
    format_specific = payload.get("format_specific_analysis", {}) or {}
    prompt_data = {
        "channel_summary": summary,
        "local_llm_analysis": llm_analysis,
        "local_action_plan": action_plan,
        "format_specific_analysis": format_specific,
    }
    return (
        "Sen ust duzey bir YouTube growth strategist, sert ama yapici bir kanal danismani ve deneyimli bir editorial director gibi dusun.\n"
        "Asagidaki kanal analytics verilerini ve local analiz ciktisini okuyup derin, net ve stratejik bir ikinci gorus hazirla.\n"
        "Bu veriler gercek kanal verisidir. Veride olmayan hicbir metriği uydurma.\n"
        "Yuzeysel konusma. Genel tavsiye verme. Mutlaka veriye dayali, acimasiz ama yapici ol.\n"
        "Tum cevabi Turkce ver.\n\n"
        "Lutfen su basliklarda cevap ver:\n"
        "1. Kanalin mevcut durumu ve buyume asamasi\n"
        "2. Son donemde ne calisiyor, ne calismiyor\n"
        "3. Icerik stratejisindeki ana sorunlar\n"
        "4. Paketleme, baslik ve thumbnail tarafinda detayli yorum\n"
        "5. Sonraki 30 gun icin somut yol haritasi\n"
        "6. Test edilmesi gereken 5 deney\n"
        "7. Uzak durulmasi gereken tekrar kaliplari\n"
        "8. Mumkunse kanal icin 10 yeni video fikri\n\n"
        "Yaniti hazirlarken sunlara ozellikle dikkat et:\n"
        "- Kendi kendini tekrar eden baslik kaliplarini acikca tespit et.\n"
        "- Buyumeyi sinirlayan en kritik 3 sorunu acikca adlandir.\n"
        "- Soylemesi zor olsa bile neyi birakmak gerektigini net soyle.\n"
        "- Eylem maddeleri uygulanabilir ve spesifik olsun.\n\n"
        "Asagidaki veriyi referans al:\n"
        f"{json.dumps(prompt_data, ensure_ascii=False, indent=2)}\n"
    )


def build_video_external_prompt(payload: dict) -> str:
    video_summary = payload.get("video_summary", {}) or {}
    retention = payload.get("retention_analysis", {}) or {}
    llm_analysis = payload.get("llm_analysis", {}) or {}
    action_plan = payload.get("critic_and_action_plan", {}) or {}
    prompt_data = {
        "video_summary": video_summary,
        "retention_analysis": retention,
        "local_llm_analysis": llm_analysis,
        "local_action_plan": action_plan,
    }
    return (
        "Sen ust duzey bir YouTube growth strategist, sert ama yapici bir video editoru ve postmortem analist gibi dusun.\n"
        "Asagidaki secili video analytics verilerini, retention ozetini ve local analiz ciktisini okuyup derin bir postmortem analiz yap.\n"
        "Bu veriler gercek video verisidir. Veride olmayan hicbir metriği uydurma.\n"
        "Yuzeysel konusma. Genel tavsiye verme. Mutlaka veriye dayali, acik ve uygulanabilir ol.\n"
        "Tum cevabi Turkce ver.\n\n"
        "Lutfen su basliklarda cevap ver:\n"
        "1. Bu video neden beklenenden iyi ya da kotu performans gostermis olabilir\n"
        "2. Goruntulenme, izlenme yuzdesi ve retention verisini birlikte yorumla\n"
        "3. En kritik 5 root cause\n"
        "4. Hook, kurgu, tempo ve payoff acisindan detayli yorum\n"
        "5. Baslik / thumbnail / paketleme tarafinda neler duzeltilmeli\n"
        "6. Sonraki video icin uygulanabilir aksiyon plani\n"
        "7. Bu videodan cikarilacak en onemli dersler\n"
        "8. Ayni konu yeniden islenecekse nasil daha iyi yapilmali\n\n"
        "Yaniti hazirlarken sunlara ozellikle dikkat et:\n"
        "- Videonun gercek basarisizlik veya basari nedenlerini yumusatmadan soyle.\n"
        "- Root cause maddeleri birbirinden farkli ve net olsun.\n"
        "- Retention sorunlarini sadece genel degil, yapisal olarak yorumla.\n"
        "- Sonraki video planini olabildigince somut yaz.\n\n"
        "Asagidaki veriyi referans al:\n"
        f"{json.dumps(prompt_data, ensure_ascii=False, indent=2)}\n"
    )


def build_legacy_feedback_report_text(payload: dict) -> str:
    retention = payload.get("retention_analysis", {}) or {}
    lines = [
        "=== ANALITIK GERI BILDIRIM DONGUSU RAPORU ===",
        "",
        retention.get("summary", "Retention verisi alinamadi."),
        "",
        "KACINILACAK KALIPLAR",
        "-" * 60,
    ]
    for item in retention.get("patterns_to_avoid", []):
        lines.append(f"- {item}")
    lines.extend(["", "HOOK URETICI ICIN NOTLAR", "-" * 60])
    for item in retention.get("hook_rewriter_guidance", []):
        lines.append(f"- {item}")
    lines.extend(["", "KESIM ONERILERI ICIN NOTLAR", "-" * 60])
    for item in retention.get("trim_suggester_guidance", []):
        lines.append(f"- {item}")
    lines.extend(["", "B-ROLL ICIN NOTLAR", "-" * 60])
    for item in retention.get("broll_guidance", []):
        lines.append(f"- {item}")
    return "\n".join(lines).strip() + "\n"
