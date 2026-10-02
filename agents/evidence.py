"""
agents/evidence.py
──────────────────
Agent 5: Evidence Extraction

Responsibilities:
  1. Find specific log events that support the RCA hypothesis
  2. Find counter-evidence (events that challenge the hypothesis)
  3. Link each evidence item to the hypothesis
  4. Build the causal graph data (nodes + edges) for visualization
"""

import json
import re
from typing import List

from core.state import IncidentState, EvidenceItem
from core.llm import get_llm


def evidence_agent(state: IncidentState) -> IncidentState:
    """
    Extracts evidence supporting (and challenging) the RCA hypothesis.
    Returns updated state.
    """
    print("[Evidence] Extracting supporting evidence...")

    hypothesis = state["rca_hypothesis"]
    timeline   = state["timeline"]

    if not hypothesis or not timeline:
        return {**state, "evidence": [], "counter_evidence": []}

    evidence, counter_evidence = _extract_evidence_with_llm(
        hypothesis, timeline, state["llm_provider"]
    )

    print(f"[Evidence] Found {len(evidence)} supporting, {len(counter_evidence)} counter-evidence items")

    return {
        **state,
        "evidence":         evidence,
        "counter_evidence": counter_evidence,
    }


def _extract_evidence_with_llm(
    hypothesis: str,
    timeline: List,
    llm_provider: str
) -> tuple:
    """Ask the LLM to identify which timeline events support/challenge the hypothesis."""

    timeline_text = "\n".join([
        f"{e['order']}. {e['timestamp']} | {e['component']} | {e['event']} — {e['note']}"
        for e in timeline
    ])

    prompt = f"""You are extracting evidence for a post-incident report.

ROOT CAUSE HYPOTHESIS:
"{hypothesis}"

TIMELINE EVENTS:
{timeline_text}

Task: Identify which events support this hypothesis and which challenge it.

Return ONLY valid JSON with no other text:
{{
  "supporting_evidence": [
    {{
      "id": 1,
      "log_entry": "exact event text from the timeline",
      "relevance": "one sentence explaining how this supports the hypothesis"
    }}
  ],
  "counter_evidence": [
    "One sentence describing a timeline event that doesn't fit the hypothesis"
  ]
}}

Rules:
- Only include evidence that is directly relevant
- counter_evidence should be honest — it strengthens the report
- If no counter_evidence exists, return an empty array
- Maximum 5 supporting evidence items, 3 counter_evidence items"""

    try:
        llm = get_llm(llm_provider)
        response = llm.invoke(prompt)
        content  = response.content.strip()

        content = re.sub(r'^```(?:json)?\s*', '', content)
        content = re.sub(r'\s*```$', '', content)

        data = json.loads(content)

        evidence = [EvidenceItem(**e) for e in data.get("supporting_evidence", [])]
        counter  = data.get("counter_evidence", [])

        return (evidence, counter)

    except Exception as e:
        print(f"[Evidence] LLM extraction failed: {e}")
        return ([], [])
