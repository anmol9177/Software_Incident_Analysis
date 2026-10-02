"""
agents/rca.py
─────────────
Agent 3: Root Cause Analysis (RCA)

This is the most important agent in the system.

Responsibilities:
  1. Analyze the reconstructed timeline
  2. Generate a primary root cause hypothesis
  3. Generate alternative hypotheses
  4. Compute a MATHEMATICALLY GROUNDED confidence score
     (3-factor model: evidence coverage + temporal coherence + LLM self-score)

The 3-factor confidence model is one of the project's novel contributions.
It transforms a vague LLM "I'm 82% confident" into something defensible.
"""

import json
import re
from typing import List

from core.state import IncidentState, TimelineEntry, ConfidenceBreakdown
from core.llm import get_llm


# ── Confidence score weights ───────────────────────────────────────────────────
# These are the w1, w2, w3 from the master plan
W_EVIDENCE   = 0.40   # How much supporting evidence exists
W_TEMPORAL   = 0.30   # Does the cause precede all effects?
W_LLM_SELF   = 0.30   # How confident is the LLM in its own answer?


def rca_agent(state: IncidentState) -> IncidentState:
    """
    Generates root cause hypothesis and computes confidence score.
    Returns updated state.
    """
    print(f"[RCA] Running analysis (iteration {state['iteration_count'] + 1})...")

    timeline = state["timeline"]

    if not timeline:
        return {
            **state,
            "rca_hypothesis": "Insufficient data — no timeline events available.",
            "rca_confidence": 0.0,
            "errors": state["errors"] + ["RCA: empty timeline"]
        }

    # Step 1: Generate hypothesis using LLM
    hypothesis, alternatives, llm_self_score = _generate_hypothesis(
        timeline,
        state["reflection_notes"],   # empty on first run, populated after reflection
        state["llm_provider"]
    )

    # Step 2: Compute confidence score (the mathematical model)
    confidence_breakdown = _compute_confidence(
        hypothesis=hypothesis,
        timeline=timeline,
        critical_count=state["critical_count"],
        llm_self_score=llm_self_score
    )

    print(f"[RCA] Hypothesis: {hypothesis[:80]}...")
    print(f"[RCA] Confidence: {confidence_breakdown['final_score']:.1%} "
          f"(evidence={confidence_breakdown['evidence_coverage']:.2f}, "
          f"temporal={confidence_breakdown['temporal_coherence']:.2f}, "
          f"llm={confidence_breakdown['llm_self_score']:.2f})")

    return {
        **state,
        "rca_hypothesis":         hypothesis,
        "rca_confidence":         confidence_breakdown["final_score"],
        "confidence_breakdown":   confidence_breakdown,
        "alternative_hypotheses": alternatives,
        "iteration_count":        state["iteration_count"] + 1,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _generate_hypothesis(
    timeline: List[TimelineEntry],
    reflection_notes: str,
    llm_provider: str
) -> tuple:
    """
    Asks the LLM to generate a root cause hypothesis.

    Returns:
        (primary_hypothesis, alternative_hypotheses, llm_self_confidence)
    """
    timeline_text = "\n".join([
        f"{e['order']}. [{e['severity'].upper()}] {e['timestamp']} | "
        f"{e['component']} | {e['event']} — {e['note']}"
        for e in timeline
    ])

    reflection_section = ""
    if reflection_notes:
        reflection_section = f"""
Previous analysis was flagged for improvement. Reviewer's notes:
{reflection_notes}

Address these concerns in your new hypothesis.
"""

    prompt = f"""You are a senior SRE performing root cause analysis on a software incident.

INCIDENT TIMELINE:
{timeline_text}
{reflection_section}

Your task: Identify the most probable root cause of this incident.

Critical rules:
- Correlation ≠ Causation. Only state a cause if evidence supports it.
- The root cause is what TRIGGERED the incident, not a symptom.
- Consider cascading failures — the visible error may not be the root cause.

Return ONLY a valid JSON object with NO extra text:
{{
  "primary_hypothesis": "One clear, specific sentence describing the root cause.",
  "supporting_reasoning": "2-3 sentences explaining why this is the root cause based on the timeline.",
  "alternative_hypotheses": ["Alternative 1 in one sentence", "Alternative 2 in one sentence"],
  "llm_confidence": 0.82
}}

Where llm_confidence is your honest self-assessment of how confident you are (0.0-1.0).
Be conservative — if evidence is weak, say 0.4-0.5, not 0.9."""

    llm = get_llm(llm_provider)
    response = llm.invoke(prompt)
    content  = response.content.strip()

    # Strip markdown fences
    content = re.sub(r'^```(?:json)?\s*', '', content)
    content = re.sub(r'\s*```$', '', content)

    try:
        data = json.loads(content)
        hypothesis   = data.get("primary_hypothesis", "Unable to determine root cause.")
        alternatives = data.get("alternative_hypotheses", [])
        llm_score    = float(data.get("llm_confidence", 0.5))
        return (hypothesis, alternatives, llm_score)
    except Exception as e:
        print(f"[RCA] Failed to parse LLM response: {e}")
        return ("Unable to determine root cause — LLM parsing failed.", [], 0.0)


def _compute_confidence(
    hypothesis: str,
    timeline: List[TimelineEntry],
    critical_count: int,
    llm_self_score: float
) -> ConfidenceBreakdown:
    """
    Computes the 3-factor confidence score.

    Factor 1: Evidence Coverage
      = proportion of critical events that mention keywords from the hypothesis

    Factor 2: Temporal Coherence
      = 1.0 if a 'critical' event precedes 'warning'/'info' events
        0.5 if unclear
        0.0 if no causal ordering can be established

    Factor 3: LLM Self-Score
      = the LLM's own stated confidence (0.0–1.0)
    """

    # ── Factor 1: Evidence Coverage ──────────────────────────────────────────
    # Extract keywords from hypothesis (crude but effective)
    keywords = set(re.findall(r'\b[a-zA-Z]{4,}\b', hypothesis.lower()))
    keywords -= {"this", "that", "with", "from", "have", "been", "were", "they", "their", "most", "likely"}

    if critical_count == 0 or not timeline:
        evidence_coverage = 0.0
    else:
        supporting = 0
        for entry in timeline:
            entry_text = (entry["event"] + " " + entry["note"]).lower()
            if any(kw in entry_text for kw in keywords):
                supporting += 1
        evidence_coverage = min(supporting / max(critical_count, 1), 1.0)

    # ── Factor 2: Temporal Coherence ─────────────────────────────────────────
    severities = [e["severity"] for e in timeline]
    if not severities:
        temporal_coherence = 0.5
    else:
        # Good sign: critical events appear before warning/info events
        first_critical_idx = next((i for i, s in enumerate(severities) if s == "critical"), None)
        if first_critical_idx is None:
            temporal_coherence = 0.5   # no critical events, uncertain
        elif first_critical_idx == 0:
            temporal_coherence = 1.0   # critical event is first — clear trigger
        elif first_critical_idx < len(severities) // 2:
            temporal_coherence = 0.75  # critical event in first half
        else:
            temporal_coherence = 0.4   # critical event late — may be a symptom, not cause

    # ── Factor 3: LLM Self-Score ──────────────────────────────────────────────
    # Clamp to [0, 1]
    llm_self_score_clamped = max(0.0, min(1.0, llm_self_score))

    # ── Weighted Final Score ──────────────────────────────────────────────────
    final_score = (
        W_EVIDENCE * evidence_coverage +
        W_TEMPORAL * temporal_coherence +
        W_LLM_SELF * llm_self_score_clamped
    )

    return ConfidenceBreakdown(
        evidence_coverage=round(evidence_coverage, 3),
        temporal_coherence=round(temporal_coherence, 3),
        llm_self_score=round(llm_self_score_clamped, 3),
        final_score=round(final_score, 3)
    )


# ── Routing function (used by LangGraph) ─────────────────────────────────────

def should_reflect_or_proceed(state: IncidentState) -> str:
    """
    Conditional routing function for LangGraph.

    Returns:
        "reflect"         → if confidence is below threshold AND under max loops
        "extract_evidence" → if confidence is acceptable OR max loops reached
    """
    import os
    threshold  = float(os.getenv("CONFIDENCE_THRESHOLD", 0.70))
    max_loops  = int(os.getenv("MAX_REFLECTION_LOOPS", 3))

    confidence = state["rca_confidence"]
    iterations = state["iteration_count"]

    if confidence < threshold and iterations < max_loops:
        print(f"[RCA Router] Confidence {confidence:.1%} < {threshold:.0%} — routing to Reflection Agent")
        return "reflect"
    else:
        print(f"[RCA Router] Confidence {confidence:.1%} — proceeding to Evidence Extraction")
        return "extract_evidence"
