"""
core/graph.py
─────────────
The LangGraph StateGraph — the brain of the entire system.

This file wires all agents together with:
  - Nodes (each agent is a node)
  - Edges (the flow between agents)
  - Conditional edges (the feedback loop — what makes this AGENTIC)

Graph flow:
  START
    │
    ▼
  log_parser ──────────────────────────────┐
    │                                       │
    ▼                                       │
  timeline                                 │
    │                                       │
    ▼                                       │
  rca_agent ◄──────────────────────────────┘
    │                                       ▲
    ├── confidence >= threshold ──► evidence │
    │                                        │
    └── confidence < threshold  ──► reflect ─┘
                                    (loops back, max 3x)
  evidence
    │
    ▼
  reporter
    │
    ▼
  END
"""

from langgraph.graph import StateGraph, END

from core.state import IncidentState
from agents.log_parser  import log_parser_agent
from agents.timeline    import timeline_agent
from agents.rca         import rca_agent, should_reflect_or_proceed
from agents.reflection  import reflection_agent
from agents.evidence    import evidence_agent
from agents.reporter    import reporter_agent


def build_graph() -> StateGraph:
    """
    Constructs and compiles the LangGraph StateGraph.

    Returns:
        A compiled LangGraph app ready to invoke.
    """

    # ── Create the graph ──────────────────────────────────────────────────────
    graph = StateGraph(IncidentState)

    # ── Register all agent nodes ──────────────────────────────────────────────
    graph.add_node("log_parser",       log_parser_agent)
    graph.add_node("timeline",         timeline_agent)
    graph.add_node("rca_agent",        rca_agent)
    graph.add_node("reflection_agent", reflection_agent)
    graph.add_node("evidence_agent",   evidence_agent)
    graph.add_node("reporter",         reporter_agent)

    # ── Set the entry point ───────────────────────────────────────────────────
    graph.set_entry_point("log_parser")

    # ── Add sequential edges ──────────────────────────────────────────────────
    graph.add_edge("log_parser", "timeline")
    graph.add_edge("timeline",   "rca_agent")

    # ── THE KEY: Conditional edge from RCA (this is what makes it agentic) ───
    graph.add_conditional_edges(
        "rca_agent",                    # from this node
        should_reflect_or_proceed,      # call this function to decide
        {
            "reflect":          "reflection_agent",   # low confidence → reflect
            "extract_evidence": "evidence_agent",     # good confidence → continue
        }
    )

    # ── Reflection loops back to RCA ──────────────────────────────────────────
    graph.add_edge("reflection_agent", "rca_agent")

    # ── Final sequential edges ────────────────────────────────────────────────
    graph.add_edge("evidence_agent", "reporter")
    graph.add_edge("reporter",       END)

    # ── Compile and return ────────────────────────────────────────────────────
    return graph.compile()


# ── Convenience function ──────────────────────────────────────────────────────

def run_analysis(raw_logs: str, incident_name: str, llm_provider: str = "groq") -> dict:
    """
    Run the full agentic pipeline on a set of logs.

    Args:
        raw_logs:      Raw log text (string)
        incident_name: Human-readable name for this incident
        llm_provider:  "groq" | "ollama"

    Returns:
        The final IncidentState dict (containing the full report)
    """
    from core.state import initial_state

    app   = build_graph()
    state = initial_state(raw_logs, incident_name, llm_provider)

    print(f"\n{'='*60}")
    print(f"  IncidentSight Analysis: {incident_name}")
    print(f"  LLM Provider: {llm_provider}")
    print(f"{'='*60}\n")

    final_state = app.invoke(state)

    print(f"\n{'='*60}")
    print(f"  Analysis complete!")
    print(f"  Root cause: {final_state['rca_hypothesis'][:80]}...")
    print(f"  Confidence: {final_state['rca_confidence']:.1%}")
    print(f"  Iterations: {final_state['iteration_count']}")
    print(f"{'='*60}\n")

    return final_state
