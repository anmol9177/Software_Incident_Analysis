"""
agents/timeline.py
──────────────────
Agent 2: Timeline Reconstruction

Responsibilities:
  1. Sort critical events chronologically
  2. Remove noise (INFO events unless significant)
  3. Annotate each event with its role in the incident
  4. Identify incident start/end window
  5. Detect causal chain candidates

LLM is used for annotation — it understands the semantic meaning
of each event and adds a human-readable note.
"""

import json
import re
from typing import List

from core.state import IncidentState, TimelineEntry, LogEvent
from core.llm import get_llm

def timeline_agent(state: IncidentState) -> IncidentState:
    """
    Reconstructs an ordered incident timeline from parsed events.
    Returns updated state.
    """
    print("[Timeline] Reconstructing incident timeline...")

    critical_events = state["critical_events"]

    if not critical_events:
        print("[Timeline] No critical events found — using all parsed events")
        critical_events = state["parsed_events"][:30]  # fallback

    # Step 1: Sort events by timestamp
    sorted_events = _sort_events(critical_events)

    # Step 2: Ask LLM to annotate and build timeline
    timeline = _build_timeline_with_llm(sorted_events, state["llm_provider"])

    # Step 3: Extract incident window
    start, end, duration = _extract_incident_window(sorted_events)

    print(f"[Timeline] Built {len(timeline)} timeline entries | Window: {start} → {end}")

    return {
        **state,
        "timeline":          timeline,
        "incident_start":    start,
        "incident_end":      end,
        "incident_duration": duration,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _sort_events(events: List[LogEvent]) -> List[LogEvent]:
    """Sort events by timestamp string. Works for ISO and common formats."""
    def sort_key(e):
        return e.get("timestamp", "")
    return sorted(events, key=sort_key)


def _build_timeline_with_llm(events: List[LogEvent], llm_provider: str) -> List[TimelineEntry]:
    """
    Sends critical events to the LLM and asks it to:
    1. Annotate each event's role in the incident
    2. Mark severity as critical / warning / info
    3. Identify potential causal relationships
    """
    if not events:
        return []

    # Format events for the prompt (keep it concise)
    events_text = "\n".join([
        f"{i+1}. [{e['level']}] {e['timestamp']} | {e['component']} | {e['message']}"
        for i, e in enumerate(events[:25])  # cap at 25 to avoid token overflow
    ])

    prompt = f"""You are an expert Site Reliability Engineer performing post-incident analysis.

Here are the critical log events from a software incident, in chronological order:

{events_text}

Your task: Build a structured incident timeline.

For each event, return a JSON array with objects containing:
- "order": integer (1, 2, 3...)
- "timestamp": string (from the log)
- "component": string (service/component name)
- "event": string (concise 1-line description of what happened)
- "severity": "critical" | "warning" | "info"
- "note": string (1 sentence — what role does this play in the incident? E.g. "This is the triggering event", "Cascading failure from event 2", "Amplifies the problem")

Rules:
- Be precise and factual — only describe what the logs show
- Identify the TRIGGERING event (root of the chain)
- Mark cascading failures clearly
- Return ONLY valid JSON array, no other text"""

    try:
        llm = get_llm(llm_provider)
        response = llm.invoke(prompt)
        content  = response.content.strip()

        # Strip markdown fences
        content = re.sub(r'^```(?:json)?\s*', '', content)
        content = re.sub(r'\s*```$', '', content)

        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [TimelineEntry(**entry) for entry in parsed]

    except Exception as e:
        print(f"[Timeline] LLM annotation failed: {e}")
        # Fallback: build basic timeline from events without annotation
        return _build_basic_timeline(events)

    return []


def _build_basic_timeline(events: List[LogEvent]) -> List[TimelineEntry]:
    """Fallback: build a simple timeline without LLM annotation."""
    timeline = []
    for i, e in enumerate(events):
        timeline.append(TimelineEntry(
            order=i + 1,
            timestamp=e["timestamp"],
            component=e["component"],
            event=e["message"],
            severity="critical" if e["level"] == "ERROR" else "warning" if e["level"] == "WARN" else "info",
            note=""
        ))
    return timeline


def _extract_incident_window(events: List[LogEvent]) -> tuple:
    """
    Extracts start time, end time, and duration of the incident.
    Returns (start_str, end_str, duration_str).
    """
    if not events:
        return ("unknown", "unknown", "unknown")

    timestamps = [e["timestamp"] for e in events if e.get("timestamp")]
    if not timestamps:
        return ("unknown", "unknown", "unknown")

    start = timestamps[0]
    end   = timestamps[-1]

    # Simple duration calculation for ISO timestamps
    try:
        from datetime import datetime
        fmt = "%Y-%m-%d %H:%M:%S"
        # Try a few common formats
        for f in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"]:
            try:
                t_start = datetime.strptime(start[:19], f[:len(f)])
                t_end   = datetime.strptime(end[:19], f[:len(f)])
                delta   = t_end - t_start
                mins    = int(delta.total_seconds() // 60)
                secs    = int(delta.total_seconds() % 60)
                duration = f"{mins}m {secs}s"
                return (start, end, duration)
            except ValueError:
                continue
    except Exception:
        pass

    return (start, end, "unknown")
