from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

from topic_selection_engine.models import RawSignal


def _days_ago(days: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)


def _youtube_signals() -> List[RawSignal]:
    return [
        RawSignal(
            source="youtube",
            source_id="yt-001",
            title="Almanya'da 2500 Euro maas yeter mi? 2026 gider analizi",
            text="Turkler Almanya'da maas, kira, market ve fatura dengesiyle nasil geciniyor sorusu tekrar yukseliyor.",
            url="https://youtube.example/mock/yt-001",
            published_at=_days_ago(2),
            metadata={"views": 42000, "comments": 380},
        ),
        RawSignal(
            source="youtube",
            source_id="yt-002",
            title="Almanya'da ev bulmak neden bu kadar zor?",
            text="Berlin, Hamburg ve NRW icin kira, SCHUFA ve ev sahibi sureci hakkinda izleyici yorumlari yogun.",
            url="https://youtube.example/mock/yt-002",
            published_at=_days_ago(5),
            metadata={"views": 37000, "comments": 290},
        ),
        RawSignal(
            source="youtube",
            source_id="yt-003",
            title="Auslanderbehorde randevu krizi: oturum uzatma bekleyenler ne yapmali",
            text="Burokrasi, oturum uzatma, randevu ve belge eksigi konularinda korku ve belirsizlik baskin.",
            url="https://youtube.example/mock/yt-003",
            published_at=_days_ago(3),
            metadata={"views": 33000, "comments": 420},
        ),
        RawSignal(
            source="youtube",
            source_id="yt-004",
            title="Almanya'da is bulmak icin Almanca sart mi?",
            text="Yeni gelenler, mavi yaka isler, ofis isleri ve dil seviyesi arasindaki gerilim ilgi cekiyor.",
            url="https://youtube.example/mock/yt-004",
            published_at=_days_ago(10),
            metadata={"views": 28000, "comments": 240},
        ),
    ]


def _reddit_signals() -> List[RawSignal]:
    return [
        RawSignal(
            source="reddit",
            source_id="rd-001",
            title="Is 2500 net enough in Germany for a family?",
            text="People compare rent, kindergarten, transport and food in Germany. Turkish users ask if moving still makes sense.",
            url="https://reddit.example/mock/rd-001",
            published_at=_days_ago(1),
            metadata={"upvotes": 190, "comments": 88},
        ),
        RawSignal(
            source="reddit",
            source_id="rd-002",
            title="How do you survive Auslanderbehorde delays?",
            text="Appointment delays, residence permit anxiety, paperwork confusion and fear of losing job offers.",
            url="https://reddit.example/mock/rd-002",
            published_at=_days_ago(4),
            metadata={"upvotes": 160, "comments": 72},
        ),
        RawSignal(
            source="reddit",
            source_id="rd-003",
            title="Turkish student in Germany: rent vs dorm vs WG",
            text="Students compare dormitory, WG and solo rent. Questions focus on deposits, city choice and hidden costs.",
            url="https://reddit.example/mock/rd-003",
            published_at=_days_ago(6),
            metadata={"upvotes": 120, "comments": 40},
        ),
        RawSignal(
            source="reddit",
            source_id="rd-004",
            title="Can I find work in Germany with A2 German?",
            text="Job seekers discuss warehouse jobs, gastronomy, office work and whether English-only jobs are realistic.",
            url="https://reddit.example/mock/rd-004",
            published_at=_days_ago(8),
            metadata={"upvotes": 140, "comments": 54},
        ),
    ]


def _channel_signals() -> List[RawSignal]:
    return [
        RawSignal(
            source="channel",
            source_id="ch-001",
            title="Almanya'da yasam maliyeti: market, kira, fatura gercegi",
            text="This older channel video performed well because it used direct comparison and real monthly budget numbers.",
            url="https://channel.example/mock/ch-001",
            published_at=_days_ago(120),
            metadata={"views": 91000, "ctr": 7.8, "avg_view_duration": 46.0},
        ),
        RawSignal(
            source="channel",
            source_id="ch-002",
            title="Almanya'da ilk 30 gun: anmeldung, banka, sigorta",
            text="Bureaucracy starter guide with practical steps. Strong comments from newcomers.",
            url="https://channel.example/mock/ch-002",
            published_at=_days_ago(75),
            metadata={"views": 68000, "ctr": 6.9, "avg_view_duration": 42.0},
        ),
        RawSignal(
            source="channel",
            source_id="ch-003",
            title="Almanya'da deprem gibi kira artisi? Gercek durum ne",
            text="A recent housing piece already covered rent pressure, so novelty should drop for very similar ideas.",
            url="https://channel.example/mock/ch-003",
            published_at=_days_ago(18),
            metadata={"views": 23000, "ctr": 5.1, "avg_view_duration": 31.0},
        ),
        RawSignal(
            source="channel",
            source_id="ch-004",
            title="Almanya'da is bulmak: hangi sehirde sans daha yuksek",
            text="Jobs and city comparison topic did well with workers and new arrivals.",
            url="https://channel.example/mock/ch-004",
            published_at=_days_ago(210),
            metadata={"views": 74000, "ctr": 7.1, "avg_view_duration": 44.0},
        ),
    ]


def collect_mock_signals(since_days: int = 14, limit: int = 200) -> List[RawSignal]:
    # TODO: replace these mocks with real API adapters for YouTube, Reddit and channel analytics.
    cutoff = _days_ago(since_days)
    signals = _youtube_signals() + _reddit_signals() + _channel_signals()
    filtered = [signal for signal in signals if signal.published_at >= cutoff or signal.source == "channel"]
    return filtered[:limit]
