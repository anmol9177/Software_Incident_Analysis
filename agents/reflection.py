"""
agents/reflection.py
────────────────────
Agent 4: Reflection Agent

This agent is what makes the system TRULY AGENTIC.

When the RCA Agent produces a low-confidence hypothesis, this agent:
  1. Critically reviews the hypothesis
  2. Identifies what evidence is missing or weak
  3. Produces improvement notes
  4. Routes back to the RCA Agent for a better attempt

This feedback loop (RCA → Reflection → RCA) is the key differentiator
from a simple pipeline. Always mention this in your viva.

Max loops: controlled by MAX_REFLECTION_LOOPS env variable (default 3).
"""

import json
import re

from core.state import IncidentState
from core.llm import get_llm


def reflection_agent(state: IncidentState) -> IncidentState:
    """
    Critiques the current RCA hypothesis and generates improvement notes.
    Returns updated state with reflection_notes populated.
    """
    print(f"[Reflection] Critiquing hypothesis (iteration {state['iteration_count']})...")

    hypothesis  = state["rca_hypothesis"]
    confidence  = state["rca_confidence"]
    breakdown   = state["confidence_breakdown"]
    timeline    = state["timeline"]

    # Build context for the critique
    timeline_text = "\n".join([
        f"{e['order']}. {e['timestamp']} | {e['component']} | {e['event']}"
        for e in timeline[:20]
    ])

    prompt = f"""You are a critical reviewer of post-incident analysis reports.

The RCA Agent produced this hypothesis with LOW CONFIDENCE ({confidence:.0%}):

HYPOTHESIS: "{hypothesis}"

CONFIDENCE BREAKDOWN:
- Evidence Coverage: {breakdown['evidence_coverage']:.0%} (how many events support this)
- Temporal Coherence: {breakdown['temporal_coherence']:.0%} (does cause precede effects?)
- LLM Self-Score: {breakdown['llm_self_score']:.0%} (LLM's own confidence)

TIMELINE USED:
{timeline_text}

Your task: Identify EXACTLY what is wrong with this hypothesis and what the RCA Agent
should look for on its next attempt.

Return ONLY valid JSON with no other text:
{{
  "problems_identified": [
    "Problem 1: specific issue with the current hypothesis",
    "Problem 2: specific issue"
  ],
  "missing_evidence": "What evidence the RCA Agent should look for that was overlooked.",
  "suggested_focus": "Which timeline event(s) should the RCA Agent focus on more carefully.",
  "improvement_instructions": "Clear, direct instructions for the RCA Agent's next attempt. Be specific."
}}"""

    try:
        llm = get_llm(state["llm_provider"])
        response = llm.invoke(prompt)
        content  = response.content.strip()

        # Strip markdown fences
        content = re.sub(r'^```(?:json)?\s*', '', content)
        content = re.sub(r'\s*```$', '', content)

        data = json.loads(content)

        problems     = "\n".join(f"- {p}" for p in data.get("problems_identified", []))
        missing      = data.get("missing_evidence", "")
        focus        = data.get("suggested_focus", "")
        instructions = data.get("improvement_instructions", "")

        reflection_notes = (
            f"PROBLEMS WITH PREVIOUS HYPOTHESIS:\n{problems}\n\n"
            f"MISSING EVIDENCE: {missing}\n\n"
            f"SUGGESTED FOCUS: {focus}\n\n"
            f"INSTRUCTIONS FOR NEXT ATTEMPT: {instructions}"
        )

        print(f"[Reflection] Generated critique — routing back to RCA")
        print(f"[Reflection] Key issue: {data.get('suggested_focus', 'N/A')}")

    except Exception as e:
        print(f"[Reflection] LLM critique failed: {e}")
        reflection_notes = (
            "Previous hypothesis had low confidence. "
            "Please reconsider the root cause by looking at the earliest ERROR events "
            "and tracing their causal chain forward."
        )

    return {
        **state,
        "reflection_notes": reflection_notes,
        # Reset confidence so RCA knows to try again
        "rca_hypothesis": "",
        "rca_confidence": 0.0,
    }
