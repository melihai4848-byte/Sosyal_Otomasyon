# moduller/config.py
import os
from dotenv import load_dotenv
from moduller.logger import get_logger
from moduller.project_paths import BASE_DIR, INPUTS_DIR, OUTPUTS_DIR, ensure_workspace_structure

logger = get_logger("Config")

def reload_project_env(override: bool = True) -> None:
    # Ortam degiskenleri proje kokunden yuklenir.
    # .env.local dosyasi en son yuklenir; gizli anahtarlarin patch'lerde ezilmemesi icin daha guvenlidir.
    for env_path in (
        BASE_DIR / ".env",
        BASE_DIR / ".env.local",
    ):
        if env_path.exists():
            load_dotenv(env_path, override=override)

    # Calisma dizininden gelen ek .env yuklerini de kacirmama
    load_dotenv(override=override)


reload_project_env(override=False)

ensure_workspace_structure()

logger.info("Sistem yapılandırması yüklendi. Girdi ve Çıktı klasörleri hazır.")
