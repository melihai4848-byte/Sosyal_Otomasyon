YouTube Draft Upload Engine beklenen klasor yapisi:

input/youtube_uploads/
  video_adi_veya_is_kodu/
    video.mp4
    subtitle_tr.srt
    subtitle_en.srt
    subtitle_de.srt
    title.txt
    description.txt
    tags.txt
    settings.json
    thumbnail.jpg

Notlar:
- Gerekli dosyalar: video.mp4, title.txt, description.txt
- subtitle_tr.srt, subtitle_en.srt, subtitle_de.srt opsiyoneldir.
- thumbnail.jpg veya thumbnail.png opsiyoneldir.
- tags.txt ve settings.json opsiyoneldir.
- settings.json ornegi icin config/youtube_settings.example.json dosyasina bakin.
