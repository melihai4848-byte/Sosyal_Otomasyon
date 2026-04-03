# Sosyal Medya Otomasyon

Uzun videolardan altyazi, YouTube metadata, analiz raporlari ve Instagram icerikleri ureten moduler bir sosyal medya otomasyon sistemi.

Sistem terminal tabanlidir. Ana menu uzerinden:
- tek modul
- ayni gruptan coklu modul
- grup bazli toplu calisma
- full automation

senaryolari ile kullanilir.

## Neler Yapar

- Videodan veya sesten Turkce altyazi uretir
- Gramer duzeltmesi yapar ve EN/DE ceviri cikarir
- YouTube icin description, title, chapter, hashtag ve metadata paketi uretir
- Video analizi, hook, trim, B-roll ve thumbnail promptlari cikarir
- Instagram icin carousel, reels, story ve haftalik paylasim plani uretir
- Trend/topic research ve YouTube analytics analizi yapar
- YouTube draft upload, otomatik B-roll indirme, Premiere XML ve reels render araclari sunar

## Moduller

### 100 Altyazi Grubu

- `101` Altyazi Olusturucu (TR)
- `102` Gramer Duzenleyici (TR)
- `103` Altyazi Cevirmeni (EN-DE)

### 200 YouTube Grubu

- `201` Video Aciklamasi (Description) Olusturucu (TR-EN-DE)
- `202` Video Elestirmeni
- `203` B-Roll Prompt Uretici (16:9 Yatay)
- `204` Thumbnail Prompt Uretici (16:9 Yatay)
- `205` Muzik Prompt Olusturucu

### 300 Instagram Grubu

- `301` Carousel Fikir Uretici
- `302` Reels Fikir Uretici
- `303` Story Serisi Fikir Uretici
- `304` Etkilesim Planlayici

### 400 Arastirma Grubu

- `401` YouTube Trends Konu Fikirleri
- `402` YouTube Analytics Analizi

### 500 Araclar Grubu

- `501` Reels Olusturucu
- `502` YouTube Draft Upload Engine
- `503` Automatic B-Roll Downloader
- `504` Premiere Pro XML Entegrasyonu
- `505` Cikti Temizleyici

## Pipeline Sirasi

Ana pipeline `101-304` araligindaki moduller icin gecerlidir:

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

Notlar:
- `400` ve `500` gruplari bu ana pipeline’in disindadir.
- Ana menude grup secimini bos birakirsan full automation calisir.
- Coklu secimde bagimliliklar otomatik eklenir.
- Checkpoint bulunsa bile secilen moduller yeniden calistirilir; otomatik atlama yoktur.

## LLM Yapisi

Sistem iki temel rol kullanir:

- `Main LLM`: daha duzenli, structured ve kuralli isler
- `Smart LLM`: yorumlama, secim, packaging ve yaratici uretim

Modul bazli rol ve onerilen model tablosu kok dizindeki [llm_rol_tablosu.md](c:\Users\melih\Desktop\Sosyal_Medya_Otomasyon\llm_rol_tablosu.md) dosyasindan okunur.

Bu dosyada:
- bir modulin `main_enabled` ve `smart_enabled` durumunu
- `recommended_main`
- `recommended_smart`
- rol aciklamalarini

duzenleyebilirsin. Degisiklikler uygulama yeniden baslatildiginda devreye girer.

## Kurulum

### 1. Python

Python `3.10+` gerekir. `3.11+` tavsiye edilir.

### 2. Bagimliliklar

```bash
pip install -r requirements.txt
```

### 3. Sistem Bilesenleri

Pip disinda su araclar gerekir:

- `FFmpeg`
- `ffprobe`
- lokal model kullanacaksan `Ollama`

### 4. Ortam Degiskenleri

Ana ortam dosyasi proje kokundeki [.env](c:\Users\melih\Desktop\Sosyal_Medya_Otomasyon\.env) dosyasidir.

Sik kullanilan anahtarlar:

- `OLLAMA_LOCAL_SERVER`
- `GEMINI_API_KEY`
- `GROQ_API_KEY`
- `OPENROUTER_API_KEY`
- `APIFREELLM_API_KEY`
- `DEEPSEEK_API_KEY`
- `TRANSLATEGEMMA_MODEL_NAME`
- `DEFAULT_MAIN_LLM_PROVIDER`
- `DEFAULT_MAIN_LLM_MODEL`
- `DEFAULT_SMART_LLM_PROVIDER`
- `DEFAULT_SMART_LLM_MODEL`

Ek olarak ihtiyaca gore:
- YouTube OAuth / Analytics anahtarlari
- stok medya provider anahtarlari

### 5. Launcher

Proje kokundeki launcher:

- [run_automation.py](c:\Users\melih\Desktop\Sosyal_Medya_Otomasyon\run_automation.py)
- [run_automation.bat](c:\Users\melih\Desktop\Sosyal_Medya_Otomasyon\run_automation.bat)
- [run_automation.ps1](c:\Users\melih\Desktop\Sosyal_Medya_Otomasyon\run_automation.ps1)

`.venv` yoksa olusturur, gereksinimleri kontrol eder ve uygulamayi baslatir.

## Calistirma

Ana menuyu acmak icin:

```bash
python main.py
```

veya launcher ile:

```bash
python run_automation.py
```

Ana menu davranisi:
- bos secim: full automation
- `1xx`, `2xx`, `3xx`, `4xx`, `5xx`: ilgili grup
- `2xx,3xx`: coklu grup secimi
- grup icinde tek numara: tek modul
- grup icinde `201,202,203`: ayni gruptan coklu modul

## Klasor Yapisi

- `workspace/00_Inputs/`: kaynak video, ses, oauth ve manuel girdiler
- `workspace/00_Outputs/`: tum uretilen ciktilar
- `moduller/`: ana otomasyon modulleri ve ortak yardimci kodlar
- `topic_selection_engine/`: trend/topic research engine
- `uploader/`: YouTube draft upload altyapisi
- `config/`: ornek config dosyalari
- `workspace/logs/`: log dosyalari
- `workspace/state/`: ortak runtime state dosyalari
- `workspace/uploader/`: uploader input/success/failed akis klasorleri

## Cikti Yapisi

Ana group output klasorleri:

- `workspace/00_Outputs/100_Altyazı`
- `workspace/00_Outputs/200_YouTube`
- `workspace/00_Outputs/300_Instagram`
- `workspace/00_Outputs/400_Arastirma_Sonuclari`
- `workspace/00_Outputs/500_Araclar_Sonuclari`
- `workspace/00_Outputs/_json_cache`

Ornek metin ciktilari:

- `workspace/00_Outputs/200_YouTube/01_Metadata/YT-Metadata_TR.txt`
- `workspace/00_Outputs/200_YouTube/01_Metadata/YT-Metadata_EN.txt`
- `workspace/00_Outputs/200_YouTube/01_Metadata/YT-Metadata_DE.txt`
- `workspace/00_Outputs/200_YouTube/05_Analysis/YT-Video_Analysis_TR.txt`
- `workspace/00_Outputs/200_YouTube/05_Analysis/YT-Editing_Anaylsis_TR.txt`
- `workspace/00_Outputs/200_YouTube/03_B-Rolls/YT-B-Roll_Prompts.txt`
- `workspace/00_Outputs/200_YouTube/02_Thumbnails/YT-Thumbnail_Prompts.txt`
- `workspace/00_Outputs/200_YouTube/04_Musics/YT-Background_Music_Prompts.txt`
- `workspace/00_Outputs/300_Instagram/IG-Carousel_Fikirleri.txt`
- `workspace/00_Outputs/300_Instagram/IG-Reels_Fikirleri.txt`
- `workspace/00_Outputs/300_Instagram/IG-Story_Fikirleri.txt`
- `workspace/00_Outputs/300_Instagram/IG-Paylasim_Takvimi.txt`
- `workspace/00_Outputs/400_Arastirma_Sonuclari/Youtube_Trends-Konu_Fikirleri.txt`
- `workspace/00_Outputs/500_Araclar_Sonuclari/Reel_Shorts_Olusturucu_Raporu.txt`

## Ozel Notlar

- `101` ve `501` LLM’den cok `Whisper` ve `ffmpeg` agirlikli calisir.
- `103` ve metadata ceviri asamasinda `Main/Smart` disinda ayri translation modeli kullanilabilir.
- `201` modulunde Turkce ana metadata uretimi Smart LLM ile yapilir; EN/DE ceviri ayri ceviri modeliyle ilerleyebilir.
- `401` ve `402` research grubundadir; ana pipeline’in zorunlu parcasi degildir.
- `505` klasor yapisini koruyup ciktilari temizlemek icin kullanilir.

## Hizli Baslangic

1. `.env` dosyasini kok dizinde doldur
2. `pip install -r requirements.txt`
3. `python main.py`
4. Once `100` grubundan altyazi akisini dene
5. Sonra `200` ve `300` gruplarina gec
6. Rol/model ayari gerekiyorsa `llm_rol_tablosu.md` dosyasini duzenle

## Guvenlik

- `.env`, OAuth dosyalari ve API anahtarlari repoya eklenmemelidir.
- `workspace/00_Inputs/oauth/` altindaki kimlik dosyalari yerel kalmalidir.
- Runtime ciktilari ve loglar repoya commitlenmemelidir.
