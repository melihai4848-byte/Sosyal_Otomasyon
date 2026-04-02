# moduller/config.py
import os
from pathlib import Path
from dotenv import load_dotenv
from moduller.logger import get_logger

logger = get_logger("Config")

# Ana proje dizinini bul (moduller klasörünün bir üstü)
BASE_DIR = Path(__file__).resolve().parent.parent

def reload_project_env(override: bool = True) -> None:
    # Hem kok dizindeki hem moduller altindaki .env dosyalarini destekle.
    # .env.local dosyalari en son yuklenir; gizli anahtarlarin patch'lerde ezilmemesi icin daha guvenlidir.
    for env_path in (
        BASE_DIR / ".env",
        BASE_DIR / "moduller" / ".env",
        BASE_DIR / ".env.local",
        BASE_DIR / "moduller" / ".env.local",
    ):
        if env_path.exists():
            load_dotenv(env_path, override=override)

    # Calisma dizininden gelen ek .env yuklerini de kacirmama
    load_dotenv(override=override)


reload_project_env(override=False)

# Klasör Yolları
INPUTS_DIR = BASE_DIR / "00_Inputs"
OUTPUTS_DIR = BASE_DIR / "00_Outputs"

# Eğer klasörler yoksa otomatik oluştur
INPUTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

logger.info("Sistem yapılandırması yüklendi. Girdi ve Çıktı klasörleri hazır.")
