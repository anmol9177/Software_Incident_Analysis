"""
agents/reporter.py
──────────────────
Agent 6: Report Generator

Responsibilities:
  1. Assemble all agent outputs into a structured report dict
  2. Generate recommendations using the LLM
  3. Produce the final incident report (used by UI and PDF export)
"""

import json
import re
from datetime import datetime

from core.state import IncidentState
from core.llm import get_llm


def reporter_agent(state: IncidentState) -> IncidentState:
    """
    Generates the final structured post-incident report.
    Returns updated state with 'report' populated.
    """
    print("[Reporter] Generating final incident report...")

    recommendations = _generate_recommendations(state)

    report = {
        "meta": {
            "incident_name":   state["incident_name"],
            "generated_at":    datetime.now().isoformat(),
            "llm_provider":    state["llm_provider"],
            "total_events":    state["total_events"],
            "critical_events": state["critical_count"],
            "rca_iterations":  state["iteration_count"],
        },
        "summary": _build_summary(state),
        "timeline": state["timeline"],
        "root_cause": {
            "hypothesis":             state["rca_hypothesis"],
            "confidence":             state["rca_confidence"],
            "confidence_breakdown":   state["confidence_breakdown"],
            "alternative_hypotheses": state["alternative_hypotheses"],
        },
        "evidence": {
            "supporting":    state["evidence"],
            "counter":       state["counter_evidence"],
        },
        "incident_window": {
            "start":    state["incident_start"],
            "end":      state["incident_end"],
            "duration": state["incident_duration"],
        },
        "recommendations": recommendations,
        "errors_during_analysis": state["errors"],
    }

    print(f"[Reporter] Report complete — {len(recommendations)} recommendations generated")

    return {**state, "report": report}


def _build_summary(state: IncidentState) -> str:
    """One-paragraph executive summary of the incident."""
    return (
        f"On {state['incident_start']}, an incident occurred in the "
        f"{state['incident_name']} system lasting {state['incident_duration']}. "
        f"Analysis of {state['total_events']} log events identified "
        f"{state['critical_count']} critical events. "
        f"The probable root cause (confidence: {state['rca_confidence']:.0%}) is: "
        f"{state['rca_hypothesis']}"
    )


def _generate_recommendations(state: IncidentState) -> list:
    """Ask the LLM to generate actionable preventive recommendations."""

    prompt = f"""Based on this post-incident analysis, generate preventive recommendations.

ROOT CAUSE: {state['rca_hypothesis']}
CONFIDENCE: {state['rca_confidence']:.0%}

Generate 3-5 specific, actionable recommendations to prevent this incident from recurring.

Return ONLY valid JSON array:
[
  {{
    "priority": "high" | "medium" | "low",
    "action": "Specific action to take (imperative sentence)",
    "rationale": "Why this prevents recurrence (one sentence)"
  }}
]

Rules:
- Be specific — "Add connection pool monitoring" not "Monitor the system"
- Order by priority (high first)
- Recommendations must follow directly from the root cause"""

    try:
        llm = get_llm(state["llm_provider"])
        response = llm.invoke(prompt)
        content  = response.content.strip()

        content = re.sub(r'^```(?:json)?\s*', '', content)
        content = re.sub(r'\s*```$', '', content)

        return json.loads(content)

    except Exception as e:
        print(f"[Reporter] Recommendation generation failed: {e}")
        return [{"priority": "high", "action": "Manual review required", "rationale": "LLM generation failed"}]
