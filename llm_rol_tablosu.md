# LLM Rol Tablosu

Bu dosya modullerin `Main LLM`, `Smart LLM` ve onerilen model bilgileri icin duzenlenebilir kaynak dosyadir.

Kurallar:
- Hucre tablosu yerine modul bloklari kullanilir.
- `main_enabled` ve `smart_enabled` alanlarinda `Evet` veya `Hayir` kullan.
- `recommended_main` ve `recommended_smart` alanlarini `PROVIDER:model` formatinda yaz.
- Model yoksa `-` birak.
- Degisiklikler uygulamayi yeniden baslattiginda devreye girer.
- Alan isimlerini koru; sadece degerleri duzenle.

---
module: 101
title: Altyazi Olusturucu (TR)
main_enabled: Hayir
smart_enabled: Hayir
recommended_main: -
recommended_smart: -
main_summary: -
smart_summary: -
notes: Whisper kullanir, Main/Smart kullanmaz.

---
module: 102
title: Gramer Duzenleyici (TR)
main_enabled: Evet
smart_enabled: Hayir
recommended_main: OLLAMA:qwen3:14b
recommended_smart: -
main_summary: video-ozel glossary duzeltmesini, gramer, imla ve satir temizligini yapar
smart_summary: -
notes: -

---
module: 103
title: Altyazi Cevirmeni (EN-DE)
main_enabled: Hayir
smart_enabled: Hayir
recommended_main: -
recommended_smart: -
main_summary: -
smart_summary: -
notes: Main/Smart yerine ayri ceviri modeli kullanir.

---
module: 201
title: Video Aciklamasi (Description) Olusturucu (TR-EN-DE)
main_enabled: Hayir
smart_enabled: Evet
recommended_main: -
recommended_smart: DEEPSEEK:deepseek-reasoner
main_summary: -
smart_summary: description, baslik ve metadata paketini uretir
notes: -

---
module: 202
title: Video Elestirmeni
main_enabled: Evet
smart_enabled: Evet
recommended_main: GEMINI:gemini-3.1-flash-lite-preview
recommended_smart: DEEPSEEK:deepseek-reasoner
main_summary: ilk analitik taslagi ve yapisal islemeyi yapar
smart_summary: nihai analiz, yorum ve packaging cilarini yapar
notes: -

---
module: 203
title: B-Roll Prompt Uretici (16:9 Yatay)
main_enabled: Hayir
smart_enabled: Evet
recommended_main: -
recommended_smart: OLLAMA:kimi-k2.5:cloud
main_summary: -
smart_summary: sahne bazli B-roll fikirlerini ve promptlarini uretir
notes: -

---
module: 204
title: Thumbnail Prompt Uretici (16:9 Yatay)
main_enabled: Hayir
smart_enabled: Evet
recommended_main: -
recommended_smart: OLLAMA:kimi-k2.5:cloud
main_summary: -
smart_summary: ana thumbnail konseptlerini ve gorsel promptlari uretir
notes: -

---
module: 205
title: Muzik Prompt Olusturucu
main_enabled: Evet
smart_enabled: Hayir
recommended_main: OLLAMA:kimi-k2.5:cloud
recommended_smart: -
main_summary: muzik planini ve segment mantigini cikarir
smart_summary: -
notes: -

---
module: 301
title: Carousel Fikir Uretici
main_enabled: Evet
smart_enabled: Evet
recommended_main: GEMINI:gemini-3.1-flash-lite-preview
recommended_smart: DEEPSEEK:deepseek-reasoner
main_summary: ilk carousel aday havuzunu cikarir
smart_summary: en iyi adaylari secer ve final carousel paketini kurar
notes: -

---
module: 302
title: Reels Fikir Uretici
main_enabled: Evet
smart_enabled: Evet
recommended_main: GEMINI:gemini-3.1-flash-lite-preview
recommended_smart: OLLAMA:kimi-k2.5:cloud
main_summary: ilk reel aday havuzunu cikarir
smart_summary: en iyi reel adaylarini secer ve final packagingi yapar
notes: -

---
module: 303
title: Story Serisi Fikir Uretici
main_enabled: Evet
smart_enabled: Evet
recommended_main: GEMINI:gemini-3.1-flash-lite-preview
recommended_smart: OLLAMA:deepseek-v3.1:671b-cloud
main_summary: ilk story aday havuzunu cikarir
smart_summary: en iyi story adaylarini secer ve final story setini kurar
notes: -

---
module: 304
title: Etkilesim Planlayici
main_enabled: Hayir
smart_enabled: Hayir
recommended_main: -
recommended_smart: -
main_summary: -
smart_summary: -
notes: Main/Smart kullanmaz.

---
module: 401
title: YouTube Trends Konu Fikirleri
main_enabled: Hayir
smart_enabled: Evet
recommended_main: -
recommended_smart: DEEPSEEK:deepseek-reasoner
main_summary: -
smart_summary: trend sinyallerini ayiklar, konu fikirlerini skorlar ve en guclu video adaylarini cikarir
notes: Registry'de gorunmeyen Smart kullanimi buradan override edilir.

---
module: 402
title: YouTube Analytics Analizi
main_enabled: Evet
smart_enabled: Evet
recommended_main: GEMINI:gemini-3.1-flash-lite-preview
recommended_smart: DEEPSEEK:deepseek-reasoner
main_summary: kanal ve video verisinden ilk teshis ve analitik yorumu cikarir
smart_summary: nihai kritik, aksiyon plani ve stratejik oncelikleri netlestirir
notes: -

---
module: 501
title: Reels Olusturucu
main_enabled: Hayir
smart_enabled: Hayir
recommended_main: -
recommended_smart: -
main_summary: -
smart_summary: -
notes: Whisper ve ffmpeg kullanir, Main/Smart kullanmaz.

---
module: 502
title: YouTube Draft Upload Engine
main_enabled: Hayir
smart_enabled: Hayir
recommended_main: -
recommended_smart: -
main_summary: -
smart_summary: -
notes: -

---
module: 503
title: Automatic B-Roll Downloader
main_enabled: Hayir
smart_enabled: Hayir
recommended_main: -
recommended_smart: -
main_summary: -
smart_summary: -
notes: -

---
module: 504
title: Premiere Pro XML Entegrasyonu
main_enabled: Hayir
smart_enabled: Hayir
recommended_main: -
recommended_smart: -
main_summary: -
smart_summary: -
notes: -

---
module: 505
title: Cikti Temizleyici
main_enabled: Hayir
smart_enabled: Hayir
recommended_main: -
recommended_smart: -
main_summary: -
smart_summary: -
notes: -
