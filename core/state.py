"""
core/state.py
─────────────
The IncidentState TypedDict is the single shared object that flows
through every agent in the LangGraph pipeline.

Every agent reads from it and writes back to it.
Think of it as the "memory" of one full analysis run.
"""

from typing import TypedDict, List, Optional


# ── Individual data models (plain dicts for simplicity) ───────────────────────

class LogEvent(TypedDict):
    timestamp: str
    level: str          # ERROR | WARN | INFO | DEBUG
    component: str      # which service/module emitted this
    message: str
    is_critical: bool   # flagged by anomaly detector or LLM


class TimelineEntry(TypedDict):
    order: int
    timestamp: str
    component: str
    event: str
    severity: str       # critical | warning | info
    note: str           # short human-readable annotation


class EvidenceItem(TypedDict):
    id: int
    log_entry: str
    relevance: str      # why this supports the hypothesis


class Recommendation(TypedDict):
    priority: str       # high | medium | low
    action: str
    rationale: str


class ConfidenceBreakdown(TypedDict):
    evidence_coverage: float    # % of critical events supporting hypothesis
    temporal_coherence: float   # cause before effect? 0-1
    llm_self_score: float       # LLM rates its own confidence 0-1
    final_score: float          # weighted combination


# ── The main state object ─────────────────────────────────────────────────────

class IncidentState(TypedDict):
    # ── INPUT ─────────────────────────────────────────────
    raw_logs: str
    incident_name: str
    llm_provider: str           # "groq" or "ollama"

    # ── PARSING ───────────────────────────────────────────
    detected_format: str        # plain | json | syslog | apache | k8s
    parsed_events: List[LogEvent]
    critical_events: List[LogEvent]
    total_events: int
    critical_count: int

    # ── TIMELINE ──────────────────────────────────────────
    timeline: List[TimelineEntry]
    incident_start: str
    incident_end: str
    incident_duration: str

    # ── ROOT CAUSE ANALYSIS ───────────────────────────────
    rca_hypothesis: str
    rca_confidence: float                       # 0.0 – 1.0
    confidence_breakdown: ConfidenceBreakdown
    alternative_hypotheses: List[str]

    # ── EVIDENCE ──────────────────────────────────────────
    evidence: List[EvidenceItem]
    counter_evidence: List[str]     # what contradicts the hypothesis?

    # ── REFLECTION ────────────────────────────────────────
    reflection_notes: str
    iteration_count: int            # how many times RCA has looped

    # ── REPORT ────────────────────────────────────────────
    report: dict                    # final structured report

    # ── METADATA ──────────────────────────────────────────
    errors: List[str]               # non-fatal errors encountered


def initial_state(raw_logs: str, incident_name: str, llm_provider: str = "groq") -> IncidentState:
    """
    Returns a blank IncidentState with sensible defaults.
    Call this at the start of every new analysis.
    """
    return IncidentState(
        raw_logs=raw_logs,
        incident_name=incident_name,
        llm_provider=llm_provider,
        detected_format="unknown",
        parsed_events=[],
        critical_events=[],
        total_events=0,
        critical_count=0,
        timeline=[],
        incident_start="",
        incident_end="",
        incident_duration="",
        rca_hypothesis="",
        rca_confidence=0.0,
        confidence_breakdown=ConfidenceBreakdown(
            evidence_coverage=0.0,
            temporal_coherence=0.0,
            llm_self_score=0.0,
            final_score=0.0
        ),
        alternative_hypotheses=[],
        evidence=[],
        counter_evidence=[],
        reflection_notes="",
        iteration_count=0,
        report={},
        errors=[]
    )
