# YouTube Draft Upload Engine

Bu paket, hazir bir video klasorunu okuyup YouTube Data API uzerinden videoyu her zaman `private` olarak yukler.

Beklenen klasor yapisi:

```text
input/youtube_uploads/video_adi/
  video.mp4
  subtitle_tr.srt
  subtitle_en.srt
  subtitle_de.srt
  title.txt
  description.txt
  tags.txt
  settings.json
  thumbnail.jpg
```

Kisa kullanim:

```powershell
python main.py --folder input\youtube_uploads\video_adi
python main.py --batch
python main.py --watch
python main.py --folder input\youtube_uploads\video_adi --dry-run
```

Onemli guvenlik notu:
- Modül videoyu asla otomatik olarak publish etmez.
- `privacyStatus` her zaman `private` olarak zorlanir.
- `publishAt` ancak config tarafinda acikca izin verilirse kullanilir.
