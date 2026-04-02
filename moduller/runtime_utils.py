def format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    parts = []
    if hours:
        parts.append(f"{hours} sa")
    if minutes or hours:
        parts.append(f"{minutes} dk")
    parts.append(f"{secs} sn")
    return " ".join(parts)
