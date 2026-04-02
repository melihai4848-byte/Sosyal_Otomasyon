# Topic Selection Engine

Bu paket, bir sonraki YouTube videosunun hangi konu olmasi gerektigini secmek icin tasarlandi.

## Ne Yapar?

- YouTube, Reddit ve kanal verilerini toplar
- Metni temizler, alakasiz sinyalleri eleyebilir
- Topic cluster'lari cikarir
- Her cluster'i skorlar
- LLM veya deterministic fallback ile video fikirlerine cevirir
- `ranked_ideas.json` ve `weekly_topic_report.md` uretir

## Calistirma

```bash
python topic_selection_engine/main.py --since-days 14 --limit 200 --output-dir ./out
```

## Yerel Veri Dosyalari

JSON, JSONL veya CSV kabul edilir.

CLI ile gecilebilir:

```bash
python topic_selection_engine/main.py ^
  --youtube-file ./sample_data/youtube_export.json ^
  --reddit-file ./sample_data/reddit_export.json ^
  --channel-file ./sample_data/channel_export.json ^
  --output-dir ./out
```

## Beklenen Alanlar

- `title`
- `text` veya `description` veya `comments`
- `published_at`
- `url`
- `id`

Ek metadata alanlari:

- YouTube: `views`, `likes`, `comments`, `ctr`
- Reddit: `upvotes`, `comments`, `awards`
- Channel: `views`, `ctr`, `avg_view_duration`, `comments`

## TODO

- Gercek API adapter'lari
- Embedding tabanli clustering
- Kanal performans verisiyle feedback loop
- Competitor gap analizi
