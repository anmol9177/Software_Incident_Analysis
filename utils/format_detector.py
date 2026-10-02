"""
utils/format_detector.py
────────────────────────
Detects log format from raw text using regex patterns.
This runs BEFORE the LLM so it's fast and deterministic.

Supported formats:
  - json       : {"timestamp": ..., "level": ...}
  - syslog     : Jun 20 10:01:02 hostname service[pid]: msg
  - apache     : 127.0.0.1 - - [20/Jun/2026:10:01:02 +0000] "GET /api" 200
  - kubernetes : 2026-06-20T10:01:02.000Z pod/name msg
  - plain      : 2026-01-29 10:42:18 ERROR component message  (default)
"""

import re
from typing import Tuple


# ── Regex patterns for each format ────────────────────────────────────────────

PATTERNS = {
    "json": re.compile(r'^\s*\{.*"(timestamp|time|@timestamp)".*\}', re.MULTILINE | re.DOTALL),
    "syslog": re.compile(r'^[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\S+\s+\S+\[', re.MULTILINE),
    "apache": re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}.*\[.*\].*"(GET|POST|PUT|DELETE|PATCH)', re.MULTILINE),
    "kubernetes": re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s+(pod|node|container)/', re.MULTILINE),
    "plain": re.compile(r'\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}', re.MULTILINE),
}


def detect_format(raw_logs: str) -> Tuple[str, float]:
    """
    Detects the log format from raw text.

    Returns:
        (format_name, confidence)
        e.g. ("plain", 0.95) or ("json", 1.0)
    """
    sample = raw_logs[:2000]   # only check the first 2000 chars

    scores = {}
    for fmt, pattern in PATTERNS.items():
        matches = pattern.findall(sample)
        scores[fmt] = len(matches)

    if not any(scores.values()):
        return ("plain", 0.5)   # fallback

    # Pick the format with the most matches
    best_fmt = max(scores, key=scores.get)
    total    = sum(scores.values())
    confidence = scores[best_fmt] / total if total > 0 else 0.5

    return (best_fmt, round(confidence, 2))


def get_format_description(fmt: str) -> str:
    """Human-readable description for the UI."""
    descriptions = {
        "json":       "JSON structured logs",
        "syslog":     "Syslog format (RFC 5424)",
        "apache":     "Apache / Nginx access logs",
        "kubernetes": "Kubernetes pod logs",
        "plain":      "Plain text with timestamps",
    }
    return descriptions.get(fmt, "Unknown format")


if __name__ == "__main__":
    # Quick test
    sample = """
2026-01-29 10:42:18 ERROR database Timeout connecting to primary
2026-01-29 10:42:19 WARN  payment  Retrying request (attempt 1/3)
2026-01-29 10:42:20 ERROR payment  Payment transaction failed
    """
    fmt, conf = detect_format(sample)
    print(f"Detected format: {fmt} (confidence: {conf})")
    print(f"Description: {get_format_description(fmt)}")
