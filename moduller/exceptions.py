# moduller/exceptions.py


class VideoNotFoundError(Exception):
    """Girdi klasorunde islenecek video bulunamadiginda firlatilir."""


class LLMConnectionError(Exception):
    """API veya yerel LLM sunucusuna ulasilamadiginda firlatilir."""


class SrtReadError(Exception):
    """SRT dosyasi bozuk veya okunamadiginda firlatilir."""


class DependencyError(Exception):
    """Bir modul onkosulu saglanmadan tetiklenmeye calisildiginda firlatilir."""

