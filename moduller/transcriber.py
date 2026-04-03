# moduller/transcriber.py
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from moduller.hardware_profiles import DetectedHardware, detect_hardware
from moduller.logger import get_logger
from moduller.project_paths import MODELS_DIR

logger = get_logger("YZ_Transkript")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    logger.warning(f"Gecersiz {name} degeri bulundu: {raw}. Varsayilan {default} kullanilacak.")
    return default


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
        return value if value >= minimum else default
    except (TypeError, ValueError):
        logger.warning(f"Gecersiz {name} degeri bulundu: {raw}. Varsayilan {default} kullanilacak.")
        return default


def resolve_shorts_word_limit() -> int:
    return _env_int("WHISPER_SHORTS_WORD_LIMIT", 3, minimum=1)


@dataclass
class SubtitleToken:
    text: str
    start: float
    end: float


@dataclass
class SubtitleEntry:
    start: float
    end: float
    text: str


@dataclass
class BasicTranscriptionInfo:
    language: str
    language_probability: float | None = None


def _normalize_token_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _normalize_subtitle_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    cleaned = re.sub(r"\s+([,.;:!?%])", r"\1", cleaned)
    cleaned = re.sub(r"([(\[{])\s+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([)\]}])", r"\1", cleaned)
    return cleaned.strip()


def _word_count(text: str) -> int:
    return len([item for item in str(text or "").split(" ") if item.strip()])


def _ends_with_strong_boundary(text: str) -> bool:
    normalized = str(text or "").rstrip()
    return normalized.endswith((".", "!", "?", "…"))


def _ends_with_soft_boundary(text: str) -> bool:
    normalized = str(text or "").rstrip()
    return normalized.endswith((".", "!", "?", "…", ",", ";", ":"))


def _resolve_standard_segmentation_settings() -> dict:
    target_chars = _env_int("WHISPER_STANDARD_TARGET_CHARS", 72, minimum=24)
    max_chars = _env_int("WHISPER_STANDARD_MAX_CHARS", 96, minimum=36)
    if max_chars <= target_chars:
        max_chars = target_chars + 12

    min_chars = _env_int("WHISPER_STANDARD_MIN_CHARS", 28, minimum=1)
    if min_chars >= max_chars:
        min_chars = max(12, max_chars // 3)

    min_words = _env_int("WHISPER_STANDARD_MIN_WORDS", 4, minimum=1)
    max_duration_ms = _env_int("WHISPER_STANDARD_MAX_DURATION_MS", 6500, minimum=1000)
    max_gap_ms = _env_int("WHISPER_STANDARD_MAX_GAP_MS", 700, minimum=0)
    merge_max_chars = _env_int("WHISPER_STANDARD_MERGE_MAX_CHARS", 124, minimum=max_chars)
    merge_max_duration_ms = _env_int(
        "WHISPER_STANDARD_MERGE_MAX_DURATION_MS",
        max_duration_ms + 2500,
        minimum=max_duration_ms,
    )
    return {
        "enabled": _env_bool("WHISPER_STANDARD_LLM_SEGMENTATION", True),
        "target_chars": target_chars,
        "max_chars": max_chars,
        "min_chars": min_chars,
        "min_words": min_words,
        "max_duration_ms": max_duration_ms,
        "max_gap_ms": max_gap_ms,
        "merge_max_chars": merge_max_chars,
        "merge_max_duration_ms": merge_max_duration_ms,
    }


def _estimate_tokens_from_text(text: str, start: float, end: float) -> list[SubtitleToken]:
    raw_words = [item for item in str(text or "").split() if item.strip()]
    if not raw_words:
        return []

    total_duration = max(float(end) - float(start), 0.01)
    word_duration = total_duration / len(raw_words)
    tokens: list[SubtitleToken] = []

    for index, raw_word in enumerate(raw_words):
        token_start = float(start) + (index * word_duration)
        token_end = token_start + word_duration
        tokens.append(
            SubtitleToken(
                text=_normalize_token_text(raw_word),
                start=token_start,
                end=max(token_end, token_start),
            )
        )
    return tokens


def _segment_to_tokens(segment) -> list[SubtitleToken]:
    segment_start = float(getattr(segment, "start", 0.0) or 0.0)
    segment_end = float(getattr(segment, "end", segment_start) or segment_start)
    segment_text = _normalize_token_text(getattr(segment, "text", ""))
    words = getattr(segment, "words", None)

    if words:
        tokens: list[SubtitleToken] = []
        for word in words:
            token_text = _normalize_token_text(getattr(word, "word", ""))
            token_start = getattr(word, "start", None)
            token_end = getattr(word, "end", None)
            if not token_text or token_start is None or token_end is None:
                tokens = []
                break
            token_start = float(token_start)
            token_end = max(float(token_end), token_start)
            tokens.append(SubtitleToken(text=token_text, start=token_start, end=token_end))
        if tokens:
            return tokens

    return _estimate_tokens_from_text(segment_text, segment_start, segment_end)


def _build_entry_from_tokens(tokens: list[SubtitleToken]) -> SubtitleEntry | None:
    if not tokens:
        return None

    text = _normalize_subtitle_text(" ".join(token.text for token in tokens))
    if not text:
        return None

    start = float(tokens[0].start)
    end = max(float(tokens[-1].end), start)
    return SubtitleEntry(start=start, end=end, text=text)


def _should_finalize_before_append(
    current_tokens: list[SubtitleToken],
    next_token: SubtitleToken,
    settings: dict,
) -> bool:
    if not current_tokens:
        return False

    current_entry = _build_entry_from_tokens(current_tokens)
    if current_entry is None:
        return False

    current_text = current_entry.text
    current_chars = len(current_text)
    current_words = _word_count(current_text)
    current_duration_ms = int(round((current_entry.end - current_entry.start) * 1000))
    gap_ms = int(round(max(0.0, next_token.start - current_tokens[-1].end) * 1000))
    tentative_text = _normalize_subtitle_text(
        " ".join([*(token.text for token in current_tokens), next_token.text])
    )
    ready_to_break = current_chars >= settings["min_chars"] or current_words >= settings["min_words"]

    if gap_ms > settings["max_gap_ms"] and ready_to_break:
        return True
    if len(tentative_text) > settings["max_chars"] and ready_to_break:
        return True
    if current_duration_ms >= settings["max_duration_ms"] and ready_to_break:
        return True
    return False


def _should_finalize_after_append(current_tokens: list[SubtitleToken], settings: dict) -> bool:
    current_entry = _build_entry_from_tokens(current_tokens)
    if current_entry is None:
        return False

    current_text = current_entry.text
    current_chars = len(current_text)
    current_words = _word_count(current_text)
    current_duration_ms = int(round((current_entry.end - current_entry.start) * 1000))
    ready_to_break = current_chars >= settings["min_chars"] or current_words >= settings["min_words"]

    if current_chars >= settings["max_chars"]:
        return True
    if _ends_with_strong_boundary(current_text) and ready_to_break:
        return True
    if current_duration_ms >= settings["max_duration_ms"] and ready_to_break:
        return True
    if current_chars >= settings["target_chars"] and _ends_with_soft_boundary(current_text) and ready_to_break:
        return True
    if current_chars >= int(settings["target_chars"] * 1.25) and current_words >= settings["min_words"]:
        return True
    return False


def _merge_adjacent_entries(entries: list[SubtitleEntry], settings: dict) -> list[SubtitleEntry]:
    if len(entries) <= 1:
        return entries

    merged = list(entries)
    changed = True
    while changed:
        changed = False
        result: list[SubtitleEntry] = []
        index = 0

        while index < len(merged):
            current = merged[index]
            current_chars = len(current.text)
            current_words = _word_count(current.text)
            needs_merge = current_chars < settings["min_chars"] or current_words < max(2, settings["min_words"] // 2)

            if needs_merge and result:
                previous = result[-1]
                gap_ms = int(round(max(0.0, current.start - previous.end) * 1000))
                combined_text = _normalize_subtitle_text(f"{previous.text} {current.text}")
                combined_duration_ms = int(round((current.end - previous.start) * 1000))
                if (
                    gap_ms <= settings["max_gap_ms"] * 2
                    and len(combined_text) <= settings["merge_max_chars"]
                    and combined_duration_ms <= settings["merge_max_duration_ms"]
                ):
                    result[-1] = SubtitleEntry(start=previous.start, end=current.end, text=combined_text)
                    changed = True
                    index += 1
                    continue

            if needs_merge and index + 1 < len(merged):
                following = merged[index + 1]
                gap_ms = int(round(max(0.0, following.start - current.end) * 1000))
                combined_text = _normalize_subtitle_text(f"{current.text} {following.text}")
                combined_duration_ms = int(round((following.end - current.start) * 1000))
                if (
                    gap_ms <= settings["max_gap_ms"] * 2
                    and len(combined_text) <= settings["merge_max_chars"]
                    and combined_duration_ms <= settings["merge_max_duration_ms"]
                ):
                    result.append(
                        SubtitleEntry(start=current.start, end=following.end, text=combined_text)
                    )
                    changed = True
                    index += 2
                    continue

            result.append(current)
            index += 1

        merged = result

    return merged


def _build_entries_from_segments(segments: list) -> list[SubtitleEntry]:
    entries: list[SubtitleEntry] = []
    for segment in segments:
        text = _normalize_subtitle_text(getattr(segment, "text", ""))
        if not text:
            continue
        start = float(getattr(segment, "start", 0.0) or 0.0)
        end = max(float(getattr(segment, "end", start) or start), start)
        entries.append(SubtitleEntry(start=start, end=end, text=text))
    return entries


def _build_llm_optimized_entries(segments: list, settings: dict) -> list[SubtitleEntry]:
    tokens: list[SubtitleToken] = []
    for segment in segments:
        tokens.extend(_segment_to_tokens(segment))

    if not tokens:
        return _build_entries_from_segments(segments)

    entries: list[SubtitleEntry] = []
    current_tokens: list[SubtitleToken] = []

    for token in tokens:
        if not token.text:
            continue

        if _should_finalize_before_append(current_tokens, token, settings):
            entry = _build_entry_from_tokens(current_tokens)
            if entry is not None:
                entries.append(entry)
            current_tokens = []

        current_tokens.append(token)
        if _should_finalize_after_append(current_tokens, settings):
            entry = _build_entry_from_tokens(current_tokens)
            if entry is not None:
                entries.append(entry)
            current_tokens = []

    if current_tokens:
        entry = _build_entry_from_tokens(current_tokens)
        if entry is not None:
            entries.append(entry)

    return _merge_adjacent_entries(entries, settings)


def _build_dynamic_entries(segments: list, kelime_siniri: int) -> list[SubtitleEntry]:
    entries: list[SubtitleEntry] = []
    effective_limit = max(1, int(kelime_siniri))

    for segment in segments:
        tokens = _segment_to_tokens(segment)
        if not tokens:
            continue
        for index in range(0, len(tokens), effective_limit):
            entry = _build_entry_from_tokens(tokens[index:index + effective_limit])
            if entry is not None:
                entries.append(entry)

    return entries


@dataclass(frozen=True)
class WhisperModelSettings:
    model_size: str
    device: str
    compute_type: str


@dataclass(frozen=True)
class WhisperBackendSelection:
    profile_name: str
    backend_name: str
    reason: str
    hardware: DetectedHardware
    settings: WhisperModelSettings


class BaseWhisperBackend(ABC):
    def __init__(self, selection: WhisperBackendSelection):
        self.selection = selection

    @property
    def signature(self) -> tuple[str, str, str, str]:
        settings = self.selection.settings
        return (
            self.selection.backend_name,
            settings.model_size,
            settings.device,
            settings.compute_type,
        )

    @abstractmethod
    def transcribe(self, video_path: str, **options) -> tuple[Any, Any]:
        raise NotImplementedError


class FasterWhisperBackend(BaseWhisperBackend):
    def __init__(self, selection: WhisperBackendSelection):
        super().__init__(selection)
        self._model = None

    def _get_model(self):
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper kutuphanesi bulunamadi. "
                "Bu backend icin requirements kurulumu gerekli."
            ) from exc

        settings = self.selection.settings
        logger.info(
            "Whisper backend yukleniyor... "
            f"(Backend: {self.selection.backend_name}, Model: {settings.model_size}, "
            f"Cihaz: {settings.device}, Compute: {settings.compute_type})"
        )
        self._model = WhisperModel(
            settings.model_size,
            device=settings.device,
            compute_type=settings.compute_type,
        )
        return self._model

    def transcribe(self, video_path: str, **options) -> tuple[Any, Any]:
        model = self._get_model()
        return model.transcribe(video_path, **options)


class OnnxQnnWhisperBackend(BaseWhisperBackend):
    def __init__(self, selection: WhisperBackendSelection):
        super().__init__(selection)
        self._model = None
        self._processor = None
        self._provider_label = "cpu"

    def _resolve_model_dir(self) -> Path | None:
        configured_path = str(os.getenv("WHISPER_ONNX_MODEL_PATH", "") or "").strip()
        if configured_path:
            candidate = Path(configured_path).expanduser()
            if not candidate.is_absolute():
                candidate = Path(__file__).resolve().parent.parent / candidate
            return candidate

        model_id = str(os.getenv("WHISPER_ONNX_MODEL_ID", "") or "").strip()
        if not model_id:
            return None

        cache_root = str(os.getenv("WHISPER_ONNX_CACHE_DIR", "") or "").strip()
        if cache_root:
            target_root = Path(cache_root).expanduser()
            if not target_root.is_absolute():
                target_root = Path(__file__).resolve().parent.parent / target_root
        else:
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_") or "whisper_onnx"
            target_root = MODELS_DIR / safe_name

        target_root.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "WHISPER_ONNX_MODEL_ID tanimli ama huggingface_hub bulunamadi."
            ) from exc

        logger.info(f"ONNX model indiriliyor / senkronize ediliyor: {model_id}")
        downloaded_path = snapshot_download(
            repo_id=model_id,
            local_dir=str(target_root),
            local_dir_use_symlinks=False,
        )
        return Path(downloaded_path)

    def _import_onnx_genai(self):
        try:
            import onnxruntime_genai as og
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime-genai kutuphanesi bulunamadi. "
                "NPU/ONNX backend icin bu bagimlilik gerekli."
            ) from exc
        return og

    def _resolve_provider(self, og) -> tuple[str | None, str]:
        provider_override = str(os.getenv("WHISPER_ONNX_PROVIDER", "auto") or "auto").strip().lower()
        requested = provider_override if provider_override else "auto"

        candidates: list[tuple[str | None, str]]
        if requested == "qnn":
            candidates = [("QNNExecutionProvider", "qnn")]
        elif requested == "dml":
            candidates = [("DmlExecutionProvider", "dml")]
        elif requested == "cpu":
            candidates = [(None, "cpu")]
        elif requested == "cuda":
            candidates = [("CUDAExecutionProvider", "cuda")]
        else:
            candidates = [
                ("QNNExecutionProvider", "qnn"),
                ("DmlExecutionProvider", "dml"),
                (None, "cpu"),
            ]

        for provider_name, provider_label in candidates:
            if provider_label == "qnn" and not getattr(og, "is_qnn_available", lambda: False)():
                continue
            if provider_label == "dml" and not getattr(og, "is_dml_available", lambda: False)():
                continue
            if provider_label == "cuda" and not getattr(og, "is_cuda_available", lambda: False)():
                continue
            return provider_name, provider_label

        return None, "cpu"

    def _load_runtime(self):
        if self._model is not None and self._processor is not None:
            return self._model, self._processor

        og = self._import_onnx_genai()
        model_dir = self._resolve_model_dir()
        if model_dir is None:
            raise RuntimeError(
                "ONNX backend secildi ama model yolu bulunamadi. "
                "WHISPER_ONNX_MODEL_PATH veya WHISPER_ONNX_MODEL_ID tanimlanmali."
            )
        if not model_dir.exists():
            raise RuntimeError(f"ONNX model klasoru bulunamadi: {model_dir}")

        provider_name, provider_label = self._resolve_provider(og)

        def _build_runtime(active_provider_name: str | None, active_provider_label: str):
            config = og.Config(str(model_dir))
            config.clear_providers()
            if active_provider_name:
                config.append_provider(active_provider_name)
            logger.info(
                "ONNX Runtime GenAI backend yukleniyor... "
                f"(ModelDir: {model_dir}, Provider: {active_provider_label})"
            )
            model = og.Model(config)
            processor = model.create_multimodal_processor()
            return model, processor

        try:
            self._model, self._processor = _build_runtime(provider_name, provider_label)
            self._provider_label = provider_label
        except Exception as exc:
            if provider_label == "cpu":
                raise
            logger.warning(
                f"ONNX provider acilamadi ({provider_label}). CPU fallback denenecek. Detay: {exc}"
            )
            self._model, self._processor = _build_runtime(None, "cpu")
            self._provider_label = "cpu"
        return self._model, self._processor

    @staticmethod
    def _probe_duration_seconds(video_path: str) -> float:
        try:
            import av
        except ImportError:
            return 0.0
        try:
            with av.open(video_path) as container:
                if container.duration is None:
                    return 0.0
                return max(float(container.duration) / 1_000_000, 0.0)
        except Exception:
            return 0.0

    @staticmethod
    def _build_prompt(task: str, language: str | None = None) -> str:
        normalized_task = str(task or "transcribe").strip().lower()
        normalized_language = str(language or "").strip().lower()

        if normalized_task == "translate":
            return (
                "Translate the spoken audio into natural English. "
                "Return only the translated transcript text."
            )

        if normalized_language and normalized_language not in {"auto", "detect"}:
            return (
                f"Transcribe the spoken audio in {normalized_language}. "
                "Return only the transcript text."
            )

        return "Transcribe the spoken audio verbatim. Return only the transcript text."

    def transcribe(self, video_path: str, **options) -> tuple[Any, Any]:
        model, processor = self._load_runtime()
        og = self._import_onnx_genai()

        prompt = self._build_prompt(options.get("task", "transcribe"), options.get("language"))
        audios = og.Audios.open(video_path)
        inputs = processor(prompt, audios=audios)

        params = og.GeneratorParams(model)
        params.set_inputs(inputs)
        params.set_search_options(
            max_length=_env_int("WHISPER_ONNX_MAX_LENGTH", 2048, minimum=128),
        )

        generator = og.Generator(model, params)
        tokenizer_stream = processor.create_stream()
        output_parts: list[str] = []

        while not generator.is_done():
            generator.generate_next_token()
            next_tokens = generator.get_next_tokens()
            if len(next_tokens) <= 0:
                continue
            output_parts.append(tokenizer_stream.decode(int(next_tokens[0])))

        transcript_text = re.sub(r"\s+", " ", "".join(output_parts)).strip()
        if not transcript_text:
            raise RuntimeError("ONNX Runtime GenAI backend bos transkript dondurdu.")

        duration_seconds = self._probe_duration_seconds(video_path)
        segment = type(
            "OnnxTranscriptSegment",
            (),
            {
                "start": 0.0,
                "end": max(duration_seconds, 0.01),
                "text": transcript_text,
                "words": None,
            },
        )()
        info = BasicTranscriptionInfo(
            language=str(options.get("language", "auto") or "auto"),
            language_probability=None,
        )
        logger.info(
            "ONNX Runtime GenAI transkripsiyon tamamlandi. "
            f"Provider={self._provider_label} | Karakter={len(transcript_text)}"
        )
        return [segment], info


class WhisperMotor:
    """Whisper altyapisini backend-secimli ve hardware-aware olarak yoneten merkezi sinif."""
    _backend = None
    _backend_signature = None

    @classmethod
    def _env_text(cls, name: str, default: str = "") -> str:
        raw = os.getenv(name)
        if raw is None:
            return default
        return str(raw).strip()

    @classmethod
    def _resolve_backend_selection(cls) -> WhisperBackendSelection:
        hardware = detect_hardware()
        backend_override = cls._env_text("WHISPER_BACKEND", "auto").lower() or "auto"
        device_hint = cls._env_text("WHISPER_DEVICE", "cuda").lower() or "cuda"

        valid_backends = {"auto", "faster_whisper_cuda", "faster_whisper_cpu", "onnx_qnn"}
        if backend_override not in valid_backends:
            logger.warning(
                f"Gecersiz WHISPER_BACKEND degeri bulundu: {backend_override}. "
                "Varsayilan auto secimi kullanilacak."
            )
            backend_override = "auto"

        onnx_backend_ready = cls._onnx_backend_ready()

        if backend_override == "faster_whisper_cuda":
            if hardware.has_cuda_gpu:
                backend_name = "faster_whisper_cuda"
                profile_name = "desktop_cuda"
                reason = "WHISPER_BACKEND zorlamasi"
            else:
                backend_name = "faster_whisper_cpu"
                profile_name = "desktop_cpu"
                reason = "WHISPER_BACKEND=faster_whisper_cuda secildi ama CUDA algilanamadi; CPU fallback"
        elif backend_override == "faster_whisper_cpu":
            backend_name = "faster_whisper_cpu"
            profile_name = "surface_arm_cpu_fallback" if hardware.can_try_npu_stack else "desktop_cpu"
            reason = "WHISPER_BACKEND zorlamasi"
        elif backend_override == "onnx_qnn":
            if hardware.can_try_npu_stack and onnx_backend_ready:
                backend_name = "onnx_qnn"
                profile_name = "surface_arm_npu"
                reason = "WHISPER_BACKEND zorlamasi"
            else:
                backend_name = "faster_whisper_cpu"
                profile_name = "surface_arm_cpu_fallback" if hardware.can_try_npu_stack else "desktop_cpu"
                reason = (
                    "WHISPER_BACKEND=onnx_qnn secildi ama uygun cihaz/model bulunamadi; "
                    "CPU fallback"
                )
        elif device_hint == "cpu":
            backend_name = "faster_whisper_cpu"
            profile_name = "surface_arm_cpu_fallback" if hardware.can_try_npu_stack else "desktop_cpu"
            reason = "WHISPER_DEVICE=cpu"
        elif device_hint == "cuda" and hardware.has_cuda_gpu:
            backend_name = "faster_whisper_cuda"
            profile_name = "desktop_cuda"
            reason = "WHISPER_DEVICE=cuda ve NVIDIA/CUDA GPU algilandi"
        elif device_hint == "cuda":
            backend_name = "faster_whisper_cpu"
            profile_name = "surface_arm_cpu_fallback" if hardware.can_try_npu_stack else "desktop_cpu"
            reason = "WHISPER_DEVICE=cuda istendi ama CUDA algilanamadi; CPU fallback"
        elif hardware.has_cuda_gpu:
            backend_name = "faster_whisper_cuda"
            profile_name = "desktop_cuda"
            reason = "Donanim profili otomatik secimi"
        elif hardware.can_try_npu_stack and onnx_backend_ready:
            backend_name = "onnx_qnn"
            profile_name = "surface_arm_npu"
            reason = "Windows ARM64/NPU adayi cihaz algilandi ve ONNX backend hazir"
        elif hardware.can_try_npu_stack:
            backend_name = "faster_whisper_cpu"
            profile_name = "surface_arm_cpu_fallback"
            reason = (
                "Windows ARM64/NPU adayi cihaz algilandi ama ONNX model/backend hazir degil; "
                "CPU fallback"
            )
        else:
            backend_name = "faster_whisper_cpu"
            profile_name = "desktop_cpu"
            reason = "Donanim profili otomatik secimi"

        settings = cls._resolve_model_settings(backend_name=backend_name, device_hint=device_hint, hardware=hardware)
        return WhisperBackendSelection(
            profile_name=profile_name,
            backend_name=backend_name,
            reason=reason,
            hardware=hardware,
            settings=settings,
        )

    @classmethod
    def _onnx_backend_ready(cls) -> bool:
        model_path = str(os.getenv("WHISPER_ONNX_MODEL_PATH", "") or "").strip()
        model_id = str(os.getenv("WHISPER_ONNX_MODEL_ID", "") or "").strip()
        if not model_path and not model_id:
            return False
        try:
            import onnxruntime_genai as og
        except ImportError:
            return False
        return bool(getattr(og, "is_qnn_available", lambda: True)())

    @classmethod
    def _resolve_model_settings(
        cls,
        *,
        backend_name: str,
        device_hint: str,
        hardware: DetectedHardware,
    ) -> WhisperModelSettings:
        model_size = cls._env_text("WHISPER_MODEL", "large-v3") or "large-v3"
        compute_type_env = cls._env_text("WHISPER_COMPUTE_TYPE", "float16").lower() or "float16"

        if backend_name == "faster_whisper_cuda":
            return WhisperModelSettings(
                model_size=model_size,
                device="cuda",
                compute_type=compute_type_env,
            )

        if backend_name == "onnx_qnn":
            return WhisperModelSettings(
                model_size=model_size,
                device="npu",
                compute_type=compute_type_env if compute_type_env not in {"", "auto"} else "int8",
            )

        compute_type = compute_type_env
        if compute_type in {"", "auto", "float16"}:
            compute_type = "int8"
            if compute_type_env == "float16":
                logger.info(
                    "CPU profili secildigi icin WHISPER_COMPUTE_TYPE float16 yerine int8 olarak ayarlandi."
                )

        if device_hint == "cuda" and not hardware.has_cuda_gpu:
            logger.warning("CUDA bulunamadigi icin transkripsiyon backend'i CPU moduna alindi.")

        return WhisperModelSettings(
            model_size=model_size,
            device="cpu",
            compute_type=compute_type,
        )

    @classmethod
    def _resolve_transcribe_options(cls, dynamic: bool = False, task_override: str | None = None) -> dict:
        standard_llm_segmentation = (not dynamic) and _env_bool("WHISPER_STANDARD_LLM_SEGMENTATION", True)
        source_language = (os.getenv("SOURCE_LANGUAGE", "auto").strip() or "auto").lower()
        options = {
            "task": str(task_override or os.getenv("WHISPER_TASK", "transcribe")).strip() or "transcribe",
            "beam_size": _env_int("WHISPER_BEAM_SIZE", 5, minimum=1),
            "best_of": _env_int("WHISPER_BEST_OF", 5, minimum=1),
            "condition_on_previous_text": _env_bool("WHISPER_CONDITION_ON_PREVIOUS_TEXT", True),
            "word_timestamps": _env_bool(
                "WHISPER_DYNAMIC_WORD_TIMESTAMPS" if dynamic else "WHISPER_STANDARD_WORD_TIMESTAMPS",
                True if dynamic or standard_llm_segmentation else False,
            ),
            "vad_filter": _env_bool("WHISPER_VAD_FILTER", False),
        }
        if source_language not in {"", "auto", "detect"}:
            options["language"] = source_language
        if standard_llm_segmentation:
            options["word_timestamps"] = True
        if options["vad_filter"]:
            options["vad_parameters"] = {
                "min_silence_duration_ms": _env_int("WHISPER_VAD_MIN_SILENCE_MS", 2000, minimum=100)
            }
        return options

    @classmethod
    def _build_backend(cls, selection: WhisperBackendSelection) -> BaseWhisperBackend:
        if selection.backend_name in {"faster_whisper_cuda", "faster_whisper_cpu"}:
            return FasterWhisperBackend(selection)
        if selection.backend_name == "onnx_qnn":
            return OnnxQnnWhisperBackend(selection)
        raise RuntimeError(f"Desteklenmeyen transkripsiyon backend'i: {selection.backend_name}")

    @classmethod
    def get_backend(cls) -> BaseWhisperBackend:
        selection = cls._resolve_backend_selection()
        signature = (
            selection.profile_name,
            selection.backend_name,
            selection.settings.model_size,
            selection.settings.device,
            selection.settings.compute_type,
        )
        if cls._backend is None or cls._backend_signature != signature:
            cls._backend = cls._build_backend(selection)
            cls._backend_signature = signature
            logger.info(
                "Whisper backend secildi: "
                f"profile={selection.profile_name} | backend={selection.backend_name} | reason={selection.reason} | "
                f"model={selection.settings.model_size} | device={selection.settings.device} | "
                f"compute={selection.settings.compute_type}"
            )
        return cls._backend

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Saniyeyi SRT zaman formatına çevirir."""
        total_milliseconds = max(0, int(round(seconds * 1000)))
        hours, remainder = divmod(total_milliseconds, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        secs, milliseconds = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    @staticmethod
    def _write_entries(output_path: str, entries: list[SubtitleEntry]) -> None:
        with open(output_path, "w", encoding="utf-8") as handle:
            for counter, entry in enumerate(entries, start=1):
                start_time = WhisperMotor.format_timestamp(entry.start)
                end_time = WhisperMotor.format_timestamp(entry.end)
                handle.write(f"{counter}\n{start_time} --> {end_time}\n{entry.text}\n\n")

    @classmethod
    def _transcribe_to_segments(cls, video_path: str, options: dict, purpose: str) -> tuple[list, object]:
        backend = cls.get_backend()
        selection = backend.selection
        language_label = options.get("language", "auto")
        logger.info(
            f"{purpose}: {Path(video_path).name} | "
            f"Gorev={options.get('task', 'transcribe')} | Dil={language_label} | Beam={options['beam_size']} | "
            f"WordTS={options['word_timestamps']} | VAD={options['vad_filter']} | "
            f"Backend={selection.backend_name} | Profile={selection.profile_name}"
        )
        segments_generator, info = backend.transcribe(video_path, **options)
        probability = getattr(info, "language_probability", None)
        probability_text = f"{probability:.3f}" if isinstance(probability, (int, float)) else "bilinmiyor"
        logger.info(f"Transkripsiyon bilgisi: dil={info.language}, olasilik={probability_text}")
        return list(segments_generator), info

    @classmethod
    def _build_standard_entries(cls, segments: list) -> list[SubtitleEntry]:
        segmentation_settings = _resolve_standard_segmentation_settings()
        if segmentation_settings["enabled"]:
            logger.info(
                "Standart raw SRT, 102/103 modulleri icin LLM-dostu segmentasyonla hazirlaniyor. "
                f"Hedef={segmentation_settings['target_chars']} karakter | "
                f"Max={segmentation_settings['max_chars']} karakter | "
                f"Min={segmentation_settings['min_chars']} karakter | "
                f"MaxSure={segmentation_settings['max_duration_ms']} ms | "
                f"MaxBosluk={segmentation_settings['max_gap_ms']} ms"
            )
            entries = _build_llm_optimized_entries(segments, segmentation_settings)
        else:
            entries = _build_entries_from_segments(segments)

        if not entries:
            logger.warning(
                "Standart transkript icin optimize segmentasyon bos dondu; "
                "Whisper segmentleri dogrudan kullanilacak."
            )
            entries = _build_entries_from_segments(segments)

        if entries:
            average_chars = round(sum(len(entry.text) for entry in entries) / len(entries), 1)
            logger.info(
                f"Standart raw SRT bloklari hazirlandi: kaynak segment={len(segments)} | "
                f"nihai blok={len(entries)} | ortalama metin boyu={average_chars} karakter"
            )
        return entries

    @classmethod
    def generate_standard_and_dynamic_transcripts(
        cls,
        video_path: str,
        standard_output_path: str,
        dynamic_output_path: str | None = None,
        kelime_siniri: int | None = None,
    ) -> None:
        options = cls._resolve_transcribe_options(dynamic=False)
        if dynamic_output_path:
            options["word_timestamps"] = True

        segments, _info = cls._transcribe_to_segments(
            video_path,
            options,
            "Standart + Shorts ortak transcribe pass baslatiliyor",
        )

        standard_entries = cls._build_standard_entries(segments)
        cls._write_entries(standard_output_path, standard_entries)
        logger.info(f"✅ Standart Altyazı hazır: {standard_output_path}")

        if dynamic_output_path:
            dynamic_entries = _build_dynamic_entries(segments, kelime_siniri or resolve_shorts_word_limit())
            cls._write_entries(dynamic_output_path, dynamic_entries)
            logger.info(f"✅ Dinamik Altyazı hazır: {dynamic_output_path}")

    @classmethod
    def generate_standard_transcript(cls, video_path: str, output_path: str) -> None:
        """Uzun YouTube videoları için standart altyazı çıkarır."""
        options = cls._resolve_transcribe_options(dynamic=False)
        segments, _info = cls._transcribe_to_segments(video_path, options, "Standart video analiz ediliyor")
        entries = cls._build_standard_entries(segments)
        cls._write_entries(output_path, entries)
        logger.info(f"✅ Standart Altyazı hazır: {output_path}")

    @classmethod
    def generate_english_translation_transcript(cls, video_path: str, output_path: str) -> None:
        """Kaynak videodan dogrudan Whisper translate gorevi ile Ingilizce SRT uretir."""
        options = cls._resolve_transcribe_options(dynamic=False, task_override="translate")
        segments, _info = cls._transcribe_to_segments(
            video_path,
            options,
            "Whisper ile dogrudan Ingilizce ceviri SRT uretimi baslatiliyor",
        )
        entries = cls._build_standard_entries(segments)
        cls._write_entries(output_path, entries)
        logger.info(f"✅ Whisper Ingilizce Altyazı hazır: {output_path}")

    @classmethod
    def generate_dynamic_transcript(
        cls,
        video_path: str,
        output_path: str,
        kelime_siniri: int | None = None,
        segments: list | None = None,
    ) -> None:
        """Reel/Shorts videoları için her satıra configurable kelime limitiyle dinamik altyazı çıkarır."""
        effective_limit = kelime_siniri or resolve_shorts_word_limit()
        if segments is None:
            options = cls._resolve_transcribe_options(dynamic=True)
            segments, _info = cls._transcribe_to_segments(
                video_path,
                options,
                "Shorts/Reel icin uygun formatta altyazi cikariliyor",
            )

        entries = _build_dynamic_entries(segments, effective_limit)
        cls._write_entries(output_path, entries)
        logger.info(f"✅ Dinamik Altyazı hazır: {output_path}")
