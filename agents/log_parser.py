"""
agents/log_parser.py
────────────────────
Agent 1: Log Parser

Responsibilities:
  1. Detect log format (via format_detector utility)
  2. Parse raw logs into structured LogEvent dicts
  3. Run statistical anomaly detection to flag critical events
  4. Update IncidentState with parsed results

The LLM is used for semantic understanding of ambiguous log messages.
Regex + scikit-learn handle the deterministic parts (fast, cheap).
"""

import json
import re
import numpy as np
from typing import List
from sklearn.ensemble import IsolationForest

from core.state import IncidentState, LogEvent
from core.llm import get_llm
from utils.format_detector import detect_format


# ── Severity level mapping ────────────────────────────────────────────────────

LEVEL_MAP = {
    "error": "ERROR", "err": "ERROR", "fatal": "ERROR", "critical": "ERROR",
    "warn":  "WARN",  "warning": "WARN",
    "info":  "INFO",  "information": "INFO",
    "debug": "DEBUG",
}

CRITICAL_LEVELS = {"ERROR", "FATAL", "CRITICAL"}


# ── Main agent function ───────────────────────────────────────────────────────

def log_parser_agent(state: IncidentState) -> IncidentState:
    """
    Parses raw logs → structured events.
    Returns updated state.
    """
    print("[LogParser] Starting log parsing...")

    raw = state["raw_logs"]

    # Step 1: Detect format
    fmt, fmt_confidence = detect_format(raw)
    print(f"[LogParser] Detected format: {fmt} (confidence: {fmt_confidence})")

    # Step 2: Parse into structured events (regex first, LLM for hard cases)
    parsed_events = _parse_logs(raw, fmt, state["llm_provider"])

    # Step 3: Anomaly detection — flag statistically unusual events
    parsed_events = _flag_anomalies(parsed_events)

    # Step 4: Separate critical events
    critical_events = [e for e in parsed_events if e["is_critical"]]

    print(f"[LogParser] Parsed {len(parsed_events)} events, {len(critical_events)} critical")

    return {
        **state,
        "detected_format":  fmt,
        "parsed_events":    parsed_events,
        "critical_events":  critical_events,
        "total_events":     len(parsed_events),
        "critical_count":   len(critical_events),
    }

# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_logs(raw: str, fmt: str, llm_provider: str) -> List[LogEvent]:
    """
    Parse raw log string into a list of LogEvent dicts.
    Uses regex for structured formats, LLM for plain text.
    """
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]

    if fmt == "json":
        return _parse_json_logs(lines)
    elif fmt in ("plain", "kubernetes"):
        return _parse_plain_logs(lines, llm_provider)
    elif fmt == "syslog":
        return _parse_syslog_logs(lines)
    elif fmt == "apache":
        return _parse_apache_logs(lines)
    else:
        return _parse_plain_logs(lines, llm_provider)


def _parse_plain_logs(lines: List[str], llm_provider: str) -> List[LogEvent]:
    """
    Parse plain text logs using regex.
    Falls back to LLM for lines that don't match the pattern.
    """
    # Common plain log pattern: TIMESTAMP LEVEL COMPONENT MESSAGE
    pattern = re.compile(
        r'^(?P<ts>\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}(?:\.\d+)?)'
        r'\s+(?P<level>\w+)'
        r'\s+(?P<component>\S+)'
        r'\s+(?P<message>.+)$'
    )

    events: List[LogEvent] = []
    unmatched = []

    for i, line in enumerate(lines):
        m = pattern.match(line)
        if m:
            level = LEVEL_MAP.get(m.group("level").lower(), m.group("level").upper())
            events.append(LogEvent(
                timestamp=m.group("ts"),
                level=level,
                component=m.group("component"),
                message=m.group("message"),
                is_critical=(level in CRITICAL_LEVELS)
            ))
        else:
            # Keep unmatched lines for LLM batch processing
            unmatched.append((i, line))

    # If we have unmatched lines, ask the LLM to parse them in one batch
    if unmatched:
        llm_events = _llm_parse_batch(unmatched, llm_provider)
        # Insert at correct positions (simplified: append)
        events.extend(llm_events)

    return events


def _parse_json_logs(lines: List[str]) -> List[LogEvent]:
    """Parse JSON-structured logs."""
    events = []
    for line in lines:
        try:
            obj = json.loads(line)
            # Handle different JSON log schemas
            ts    = obj.get("timestamp") or obj.get("time") or obj.get("@timestamp", "")
            level = obj.get("level") or obj.get("severity") or obj.get("lvl", "INFO")
            level = LEVEL_MAP.get(str(level).lower(), str(level).upper())
            comp  = obj.get("service") or obj.get("component") or obj.get("logger", "unknown")
            msg   = obj.get("message") or obj.get("msg") or str(obj)

            events.append(LogEvent(
                timestamp=str(ts),
                level=level,
                component=str(comp),
                message=str(msg),
                is_critical=(level in CRITICAL_LEVELS)
            ))
        except json.JSONDecodeError:
            continue
    return events


def _parse_syslog_logs(lines: List[str]) -> List[LogEvent]:
    """Parse RFC-5424 syslog format."""
    pattern = re.compile(
        r'^(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})'
        r'\s+(?P<host>\S+)'
        r'\s+(?P<comp>\S+?)(?:\[\d+\])?:'
        r'\s+(?P<msg>.+)$'
    )
    events = []
    for line in lines:
        m = pattern.match(line)
        if m:
            # Syslog doesn't always have a level; infer from message content
            msg   = m.group("msg")
            level = _infer_level(msg)
            events.append(LogEvent(
                timestamp=m.group("ts"),
                level=level,
                component=m.group("comp"),
                message=msg,
                is_critical=(level in CRITICAL_LEVELS)
            ))
    return events


def _parse_apache_logs(lines: List[str]) -> List[LogEvent]:
    """Parse Apache/Nginx access log format."""
    pattern = re.compile(
        r'^(?P<ip>\S+)\s+\S+\s+\S+\s+'
        r'\[(?P<ts>[^\]]+)\]\s+'
        r'"(?P<method>\S+)\s+(?P<path>\S+)[^"]*"\s+'
        r'(?P<status>\d{3})\s+'
        r'(?P<size>\d+|-)'
    )
    events = []
    for line in lines:
        m = pattern.match(line)
        if m:
            status = int(m.group("status"))
            level  = "ERROR" if status >= 500 else "WARN" if status >= 400 else "INFO"
            events.append(LogEvent(
                timestamp=m.group("ts"),
                level=level,
                component="web-server",
                message=f"{m.group('method')} {m.group('path')} → {status}",
                is_critical=(status >= 500)
            ))
    return events


def _llm_parse_batch(unmatched: List[tuple], llm_provider: str) -> List[LogEvent]:
    """
    Ask the LLM to parse log lines that didn't match any regex.
    Sends all unmatched lines in one prompt to save API calls.
    """
    if not unmatched:
        return []

    lines_text = "\n".join([f"Line {i}: {line}" for i, line in unmatched[:20]])  # cap at 20

    prompt = f"""Parse these log lines into structured JSON.
Each line is a log event. Return a JSON array.

Log lines:
{lines_text}

Return ONLY a valid JSON array with no extra text. Each object must have:
- "timestamp": string (or "" if not found)
- "level": "ERROR"|"WARN"|"INFO"|"DEBUG"
- "component": string (service/module name or "unknown")
- "message": string (the main log message)
- "is_critical": boolean (true for ERROR/FATAL only)"""

    try:
        llm = get_llm(llm_provider)
        response = llm.invoke(prompt)
        content  = response.content.strip()

        # Strip markdown code fences if present
        content = re.sub(r'^```(?:json)?\s*', '', content)
        content = re.sub(r'\s*```$', '', content)

        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [LogEvent(**e) for e in parsed[:20]]
    except Exception as e:
        print(f"[LogParser] LLM batch parse failed: {e}")

    return []


def _flag_anomalies(events: List[LogEvent]) -> List[LogEvent]:
    """
    Use Isolation Forest to detect statistically unusual events.
    Anomalies get is_critical=True regardless of log level.

    The feature we use: rolling count of ERROR events per minute.
    """
    if len(events) < 10:
        return events  # not enough data for anomaly detection

    # Build a feature: error_count per minute bucket
    # Simplified: use event index and level as features
    features = []
    for i, e in enumerate(events):
        level_score = 3 if e["level"] == "ERROR" else 2 if e["level"] == "WARN" else 1
        features.append([i, level_score])

    X = np.array(features)
    clf = IsolationForest(contamination=0.1, random_state=42)
    labels = clf.fit_predict(X)   # -1 = anomaly, 1 = normal

    for i, (event, label) in enumerate(zip(events, labels)):
        if label == -1:
            events[i] = {**event, "is_critical": True}

    return events


def _infer_level(message: str) -> str:
    """Infer log level from message content when not explicit."""
    msg_lower = message.lower()
    if any(w in msg_lower for w in ["error", "fail", "fatal", "critical", "exception", "crash"]):
        return "ERROR"
    if any(w in msg_lower for w in ["warn", "warning", "timeout", "retry", "slow"]):
        return "WARN"
    return "INFO"
