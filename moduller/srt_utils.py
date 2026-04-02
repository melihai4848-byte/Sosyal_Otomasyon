# moduller/srt_utils.py
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from moduller.logger import get_logger

logger = get_logger("SrtUtils")
TIMING_LINE_RE = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}$")

@dataclass
class SrtBlock:
    raw: str
    index_line: Optional[str]
    timing_line: Optional[str]
    text_lines: Optional[List[str]]

    @property
    def is_processable(self) -> bool:
        return self.index_line is not None and self.timing_line is not None and self.text_lines is not None

    @property
    def id(self) -> str:
        return self.index_line.strip() if self.index_line else ""

    @property
    def text_content(self) -> str:
        if not self.is_processable or not self.text_lines:
            return ""
        return "\n".join(self.text_lines).strip()


def _is_index_line(line: str) -> bool:
    return bool(re.fullmatch(r"\d+", line.strip()))


def _is_timing_line(line: str) -> bool:
    return bool(TIMING_LINE_RE.fullmatch(line.strip()))


def _normalize_text_lines(lines: Optional[List[str]]) -> List[str]:
    if not lines:
        return [""]
    cleaned = [str(line).rstrip() for line in lines]
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    return cleaned or [""]


def _build_processable_block(source: SrtBlock, text_lines: Optional[List[str]]) -> SrtBlock:
    normalized_lines = _normalize_text_lines(text_lines)
    raw = "\n".join([source.index_line or "", source.timing_line or "", *normalized_lines]).strip()
    return SrtBlock(
        raw=raw,
        index_line=source.index_line,
        timing_line=source.timing_line,
        text_lines=normalized_lines,
    )

def read_srt_file(filepath: Path) -> str:
    """SRT dosyasını okur ve içeriğini döndürür."""
    logger.info(f"Dosya okunuyor: {filepath.name}")
    return filepath.read_text(encoding="utf-8-sig")

def write_srt_file(filepath: Path, content: str) -> None:
    """SRT içeriğini dosyaya kaydeder."""
    filepath.write_text(content.strip() + "\n", encoding="utf-8")
    logger.info(f"İşlem tamamlandı. Dosya kaydedildi: {filepath.name}")

def parse_srt_blocks(text: str) -> List[SrtBlock]:
    """Metni SRT bloklarına (SrtBlock) ayırır."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized: return []
    raw_blocks = [b for b in normalized.split("\n\n") if b.strip()]
    parsed: List[SrtBlock] = []

    for raw in raw_blocks:
        lines = raw.split("\n")
        if len(lines) < 2:
            parsed.append(SrtBlock(raw=raw, index_line=None, timing_line=None, text_lines=None))
            continue
        parsed.append(SrtBlock(raw=raw, index_line=lines[0], timing_line=lines[1], text_lines=lines[2:]))
    return parsed

def serialize_srt_blocks(blocks: List[SrtBlock]) -> str:
    """SrtBlock nesnelerini tekrar metne dönüştürür."""
    out_blocks: List[str] = []
    for b in blocks:
        if not b.is_processable:
            out_blocks.append(b.raw.strip())
            continue
        lines = [b.index_line, b.timing_line, *b.text_lines]
        out_blocks.append("\n".join(lines).strip())
    return "\n\n".join(out_blocks).strip() + "\n"

def chunk_blocks(blocks: List[SrtBlock], max_chars: int = 800) -> List[List[SrtBlock]]:
    """Blokları belirtilen karakter sınırına göre gruplar (Chunking)."""
    chunks, current_chunk, current_length = [], [], 0
    for block in blocks:
        block_len = len(block.text_content) if block.is_processable else 0
        if current_length + block_len > max_chars and current_chunk:
            chunks.append(current_chunk)  
            current_chunk, current_length = [], 0            
        current_chunk.append(block)
        current_length += block_len
    if current_chunk: chunks.append(current_chunk)
    return chunks


def _extract_candidate_text_groups(candidate_text: str) -> tuple[List[SrtBlock], List[List[str]]]:
    candidate_blocks = parse_srt_blocks(candidate_text)
    groups: List[List[str]] = []

    for block in candidate_blocks:
        if block.is_processable:
            text_lines = _normalize_text_lines(block.text_lines)
            if any(line.strip() for line in text_lines):
                groups.append(text_lines)
            continue

        lines = []
        for line in block.raw.splitlines():
            stripped = line.strip()
            if not stripped or _is_index_line(stripped) or _is_timing_line(stripped):
                continue
            lines.append(stripped)
        if lines:
            groups.append(lines)

    return candidate_blocks, groups


def enforce_srt_structure(original_blocks: List[SrtBlock], candidate_text: str) -> tuple[str, Dict[str, object]]:
    diagnostics: Dict[str, object] = {
        "original_block_count": len(original_blocks),
        "candidate_block_count": 0,
        "mode": "fallback_original",
        "metadata_repairs": 0,
        "fallback_blocks": 0,
        "leftover_text_lines": 0,
    }

    if not candidate_text.strip():
        diagnostics["fallback_blocks"] = len(original_blocks)
        return serialize_srt_blocks(original_blocks).strip(), diagnostics

    candidate_blocks, text_groups = _extract_candidate_text_groups(candidate_text)
    diagnostics["candidate_block_count"] = len(candidate_blocks)

    if not candidate_blocks:
        diagnostics["fallback_blocks"] = len(original_blocks)
        return serialize_srt_blocks(original_blocks).strip(), diagnostics

    if len(candidate_blocks) == len(original_blocks):
        repaired_blocks: List[SrtBlock] = []
        metadata_repairs = 0
        fallback_blocks = 0

        for original, candidate in zip(original_blocks, candidate_blocks):
            if not original.is_processable:
                repaired_blocks.append(original)
                continue

            candidate_lines = (
                _normalize_text_lines(candidate.text_lines)
                if candidate.is_processable
                else _normalize_text_lines(
                    [
                        line.strip()
                        for line in candidate.raw.splitlines()
                        if line.strip() and not _is_index_line(line) and not _is_timing_line(line)
                    ]
                )
            )
            if not any(line.strip() for line in candidate_lines):
                candidate_lines = _normalize_text_lines(original.text_lines)
                fallback_blocks += 1

            if candidate.index_line != original.index_line or candidate.timing_line != original.timing_line:
                metadata_repairs += 1

            repaired_blocks.append(_build_processable_block(original, candidate_lines))

        diagnostics["mode"] = "positionally_repaired"
        diagnostics["metadata_repairs"] = metadata_repairs
        diagnostics["fallback_blocks"] = fallback_blocks
        return serialize_srt_blocks(repaired_blocks).strip(), diagnostics

    mutable_groups = [list(group) for group in text_groups]
    group_index = 0
    repaired_blocks: List[SrtBlock] = []
    fallback_blocks = 0

    for original in original_blocks:
        if not original.is_processable:
            repaired_blocks.append(original)
            continue

        desired_line_count = max(1, len(original.text_lines or []))
        collected_lines: List[str] = []

        while len(collected_lines) < desired_line_count and group_index < len(mutable_groups):
            current_group = mutable_groups[group_index]
            if not current_group:
                group_index += 1
                continue

            take_count = min(desired_line_count - len(collected_lines), len(current_group))
            collected_lines.extend(current_group[:take_count])
            del current_group[:take_count]

            if not current_group:
                group_index += 1

        if not any(line.strip() for line in collected_lines):
            collected_lines = _normalize_text_lines(original.text_lines)
            fallback_blocks += 1

        repaired_blocks.append(_build_processable_block(original, collected_lines))

    leftovers: List[str] = []
    for remaining_group in mutable_groups[group_index:]:
        leftovers.extend([line for line in remaining_group if line.strip()])

    if leftovers:
        diagnostics["leftover_text_lines"] = len(leftovers)
        for idx in range(len(repaired_blocks) - 1, -1, -1):
            if repaired_blocks[idx].is_processable:
                repaired_blocks[idx] = _build_processable_block(
                    original_blocks[idx],
                    list(repaired_blocks[idx].text_lines or []) + leftovers,
                )
                break

    diagnostics["mode"] = "redistributed_text"
    diagnostics["fallback_blocks"] = fallback_blocks
    diagnostics["metadata_repairs"] = len(original_blocks)
    return serialize_srt_blocks(repaired_blocks).strip(), diagnostics
