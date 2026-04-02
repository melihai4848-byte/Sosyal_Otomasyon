# YouTube Otomasyon v2

Uzun videolardan altyazi, YouTube metadata, analiz raporlari ve Instagram icerikleri ureten moduler bir sosyal medya otomasyon sistemi.

Sistem terminal tabanlidir ve ana menu uzerinden tekil modul, grup bazli calisma veya coklu secim pipeline'i ile kullanilir.

## Neler Yapar

- Videodan veya sesten Turkce altyazi uretir
- Altyaziyi duzeltir ve EN/DE ceviri cikarir
- YouTube icin title, description, chapter, hashtag ve metadata uretir
- Video analizi, trim, hook, B-roll ve thumbnail promptlari cikarir
- Instagram icin carousel, reels, story ve paylasim plani uretir
- Trend/topic research ve analytics analizi yapar
- YouTube draft upload, B-roll download, Premiere XML gibi araclar sunar

## Grup Yapisi

- `100` Altyazi grubu: `101-103`
- `200` YouTube grubu: `201-205`
- `300` Instagram grubu: `301-304`
- `400` Arastirma grubu: `401-402`
- `500` Araclar grubu: `501-505`

## Ana Moduller

### 1xx Altyazi

- `101` Altyazi Olusturucu
- `102` Gramer Duzenleyici
- `103` Altyazi Cevirmeni

### 2xx YouTube

- `201` YouTube Metadata Olusturucu
- `202` Video Analiz Uretici
- `203` B-Roll Prompt Uretici
- `204` Thumbnail Prompt Uretici
- `205` Muzik Prompt Uretici

### 3xx Instagram

- `301` Carousel Fikir Uretici
- `302` Reels Fikir Uretici
- `303` Story Serisi Fikir Uretici
- `304` Etkilesim Planlayici

### 4xx Arastirma

- `401` YouTube Trends Konu Fikirleri
- `402` YouTube Analytics Analizi

### 5xx Araclar

- `501` Reels Olusturucu
- `502` YouTube Draft Upload Engine
- `503` Automatic B-Roll Downloader
- `504` Premiere Pro XML Entegrasyonu
- `505` Cikti Temizleyici

## Pipeline Sirasi

1. `101` subtitle
2. `102` grammar
3. `103` translation
4. `201` description
5. `202` critic
6. `203` broll
7. `204` thumbnail_main
8. `205` music
9. `301` carousel
10. `302` reels
11. `303` story
12. `304` ig_metadata

Not:
- Arastirma ve tools grubu bu ana pipeline'a dahil degildir.
- Bos secimle ana menuden calistirildiginda `1xx + 2xx + 3xx` pipeline'i birlikte calisir.

## LLM Mimarisi

Sistem iki temel rol kullanir:

- `Main LLM`: daha duzenli, structured ve kuralli isler
- `Smart LLM`: packaging, secim, yorumlama ve yaratici uretim

Bazi moduller tek model kullanir, bazilari hibrittir:

- Sadece `Main`: `102`, `205`
- Sadece `Smart`: `201`, `203`, `204`
- Hibrit `Main + Smart`: `202`, `301`, `302`, `303`

Full automation seciminde sistem LLM'leri otomatik atar ve kullaniciya tekrar sormaz.

## Kurulum

### 1. Python

Python `3.11+` tavsiye edilir.

### 2. Bagimliliklar

Repo kok dizininde:

```bash
pip install -r requirements.txt
```

### 3. Sistem Bilesenleri

Pip disinda su araclar da gerekir:

- `FFmpeg`
- `ffprobe`
- lokal model kullanilacaksa `Ollama`

### 4. Ortam Degiskenleri

Ana ayarlar `moduller/.env` icinden okunur.

Ihtiyaca gore su anahtarlar kullanilir:

- `OLLAMA_LOCAL_SERVER`
- `GEMINI_API_KEY`
- `GROQ_API_KEY`
- `OPENROUTER_API_KEY`
- `HF_API_KEY`
- `TRANSLATEGEMMA_MODEL_NAME`
- YouTube OAuth / Analytics anahtarlari
- stok medya provider anahtarlari:
  - `PEXELS_API_KEY`
  - `PIXABAY_API_KEY`
  - `FREEPIK_API_KEY`
  - `COVERR_API_KEY`

## Calistirma

Ana menuyu acmak icin:

```bash
python main.py
```

Terminalde:

- once grup secilir
- sonra tekli, coklu veya tum grup calistirilabilir
- bos secim ana pipeline'i calistirir

## Klasor Yapisi

- `00_Inputs/`: kaynak video, ses ve manuel girdiler
- `00_Outputs/`: tum uretilen ciktilar
- `moduller/`: ana otomasyon modulleri
- `topic_selection_engine/`: trend/topic research engine
- `uploader/`: YouTube draft upload altyapisi
- `logs/`: log dosyalari

## Onemli Cikti Klasorleri

### YouTube

- `00_Outputs/200_YouTube/YT-Metadata_TR.txt`
- `00_Outputs/200_YouTube/YT-Metadata_EN.txt`
- `00_Outputs/200_YouTube/YT-Metadata_DE.txt`
- `00_Outputs/200_YouTube/YT-Video_Analysis_TR.txt`
- `00_Outputs/200_YouTube/YT-Editing_Anaylsis_TR.txt`
- `00_Outputs/200_YouTube/YT-B-Roll_Prompts.txt`
- `00_Outputs/200_YouTube/YT-Thumbnail_Prompts.txt`
- `00_Outputs/200_YouTube/YT-Background_Music_Prompts.txt`

### Instagram

- `00_Outputs/300_Instagram/301_IG-Carousel_Fikirleri/*.txt`
- `00_Outputs/300_Instagram/302_IG-Reels_Fikirleri/*.txt`
- `00_Outputs/300_Instagram/303_IG-Story_Fikirleri/*.txt`
- `00_Outputs/300_Instagram/IG-Paylasim_Takvimi.txt`

### Research

- `00_Outputs/400_Arastirma_Sonuclari/`

### Cache

Cogu modul ayrica JSON cache cikarir:

- `00_Outputs/**/_json_cache/`

## Coklu Secim ve Full Automation

### Coklu Secim

Ayni grup icinde birden fazla modul secildiginde:

- bagimliliklar otomatik eklenir
- gerekiyorsa ortak pipeline akisi kullanilir
- LLM gereken adimlar icin onerilen profil ekrani gosterilir

### Full Automation

Ana menude bos secim yapildiginda:

- `1xx + 2xx + 3xx` pipeline'i birlikte calisir
- LLM secimi sorulmaz
- modullere gore otomatik model profili atanir

## Notlar

- Ollama HTTP uzerinden kullanilir; ayrica `ollama` Python paketi gerekmez.
- Bazi moduller uzun ve structured JSON cevaplar urettigi icin debug dosyalari yazabilir.
- `505` Cikti Temizleyici klasor yapisini korur, sadece icerikleri temizler.
- `502` uploader publish yapmaz; video draft/private olarak guncellenir.

## Hizli Baslangic

1. `pip install -r requirements.txt`
2. `moduller/.env` ayarlarini doldur
3. `python main.py`
4. Once `100` grubundan altyazi akisini dene
5. Sonra `200` ve `300` gruplarina gec

## Lisans / Ic Not

Bu repo proje ici kullanim mantigiyla yapilandirilmistir. Ortam degiskenleri, OAuth dosyalari ve API anahtarlari repoya eklenmemelidir.
