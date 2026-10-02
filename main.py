"""
main.py
───────
Command-line entry point for IncidentSight.

Usage:
    python main.py                                    # analyze sample log
    python main.py --log path/to/your.log             # analyze your own log
    python main.py --log your.log --provider ollama   # use Ollama instead of Groq
    python main.py --test-llm                         # verify LLM connection
"""

import argparse
import json
import sys
import os

from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="IncidentSight — Agentic Post-Incident Analysis")
    parser.add_argument("--log",      type=str, help="Path to log file",         default=None)
    parser.add_argument("--name",     type=str, help="Incident name",            default="Unnamed Incident")
    parser.add_argument("--provider", type=str, help="LLM provider: groq|ollama", default=None)
    parser.add_argument("--test-llm", action="store_true", help="Test LLM connection and exit")
    parser.add_argument("--output",   type=str, help="Save report to JSON file", default=None)
    args = parser.parse_args()

    # ── Test mode ─────────────────────────────────────────────────────────────
    if args.test_llm:
        from core.llm import test_llm_connection
        ok = test_llm_connection(args.provider)
        sys.exit(0 if ok else 1)

    # ── Load log file ──────────────────────────────────────────────────────────
    if args.log:
        log_path = args.log
    else:
        # Default to sample log
        log_path = os.path.join(
            os.path.dirname(__file__),
            "data", "sample_logs", "db_timeout_incident.log"
        )
        print(f"No --log specified. Using sample: {log_path}\n")

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            raw_logs = f.read()
    except FileNotFoundError:
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)

    # ── Run analysis ───────────────────────────────────────────────────────────
    from core.graph import run_analysis

    provider = args.provider or os.getenv("DEFAULT_PROVIDER", "groq")

    final_state = run_analysis(
        raw_logs=raw_logs,
        incident_name=args.name or os.path.basename(log_path),
        llm_provider=provider
    )

    # ── Print summary ──────────────────────────────────────────────────────────
    report = final_state.get("report", {})
    print("\n" + "─" * 60)
    print("INCIDENT REPORT SUMMARY")
    print("─" * 60)
    print(f"Incident:    {final_state['incident_name']}")
    print(f"Duration:    {final_state['incident_duration']}")
    print(f"Events:      {final_state['total_events']} total, {final_state['critical_count']} critical")
    print(f"RCA:         {final_state['rca_hypothesis']}")
    print(f"Confidence:  {final_state['rca_confidence']:.1%}")
    print(f"Iterations:  {final_state['iteration_count']}")

    print("\nTIMELINE:")
    for entry in final_state.get("timeline", [])[:10]:
        icon = "🔴" if entry["severity"] == "critical" else "🟡" if entry["severity"] == "warning" else "🟢"
        print(f"  {icon} {entry['timestamp']} | {entry['event']}")

    print("\nRECOMMENDATIONS:")
    recs = report.get("recommendations", [])
    for i, rec in enumerate(recs, 1):
        priority_icon = "‼️" if rec["priority"] == "high" else "⚠️" if rec["priority"] == "medium" else "ℹ️"
        print(f"  {priority_icon} [{rec['priority'].upper()}] {rec['action']}")

    # ── Save report ────────────────────────────────────────────────────────────
    output_path = args.output or "incident_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        # Convert TypedDict to regular dict for JSON serialization
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to: {output_path}")


if __name__ == "__main__":
    main()
