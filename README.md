# IncidentSight 🔍
### Agentic AI for Automated Post-Incident Analysis

---

## Quick Setup (Windows)

### Step 1 — Clone / create project folder
```bash
# Open Command Prompt or PowerShell in this folder
cd incident_sight
```

### Step 2 — Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```
You should see `(venv)` at the start of your terminal line.

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```
This takes 2-5 minutes the first time.

### Step 4 — Set up your API key
```bash
# Copy the example file
copy .env.example .env
```
Then open `.env` in Notepad and paste your Groq API key.
Get a free key at: https://console.groq.com

### Step 5 — Test your LLM connection
```bash
python main.py --test-llm
```
You should see: `[LLM OK] Provider: groq | Response: OK`

### Step 6 — Run your first analysis
```bash
python main.py
```
This analyzes the sample DB timeout incident in `data/sample_logs/`.

---

## Project Structure

```
incident_sight/
├── agents/              # Each AI agent lives here
│   ├── log_parser.py    # Agent 1: Parse raw logs
│   ├── timeline.py      # Agent 2: Reconstruct timeline
│   ├── rca.py           # Agent 3: Root cause analysis
│   ├── reflection.py    # Agent 4: Self-critique (feedback loop)
│   ├── evidence.py      # Agent 5: Evidence extraction
│   └── reporter.py      # Agent 6: Report generation
│
├── core/
│   ├── state.py         # IncidentState TypedDict (shared memory)
│   ├── graph.py         # LangGraph StateGraph (the orchestrator)
│   └── llm.py           # LLM factory (Groq / Ollama)
│
├── utils/
│   └── format_detector.py  # Detects log format automatically
│
├── data/sample_logs/    # Test log files
├── ui/app.py            # Streamlit UI (coming soon)
├── main.py              # CLI entry point
└── requirements.txt
```

---

## Using Ollama (offline)

1. Download Ollama from https://ollama.ai
2. Open a terminal and run: `ollama pull qwen2.5:7b`
3. In your `.env`, set `DEFAULT_PROVIDER=ollama`
4. Run: `python main.py --provider ollama`

---

## Analyzing Your Own Logs

```bash
python main.py --log path/to/your/logfile.log --name "My Incident"
```

---

## Architecture

```
START → LogParser → Timeline → RCA Agent
                                   │
                          confidence >= 70%? ──YES──► Evidence → Reporter → END
                                   │
                                  NO
                                   │
                              Reflection Agent
                                   │
                          (loops back, max 3x)
```

The feedback loop (RCA → Reflection → RCA) is what makes this
a **truly agentic system**, not just a pipeline.
