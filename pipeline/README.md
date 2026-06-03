# Pipeline — MultiHint-9K Data Generation

This directory contains the multi-agent pipeline that generates and verifies the MultiHint-9K dataset.

## Architecture

The pipeline runs two phases per Wikipedia passage:

### Phase 1: QA Generation
`Generator → QA Critic → QA Refiner (up to N=2 loops)`

### Phase 2: Hint Scaffolding
`Hint Architect → Hint Critic → Hint Refiner (up to N=2 loops) → Recovery Agent`

## Setup

```bash
# Install dependencies
pip install openai google-generativeai mistralai python-dotenv

# Create your .env file
cp .env.example .env
# Fill in your API keys
```

## Configuration

Edit `pipeline_orchestrator.py` to set:
- `DATA_FILES` — input CSV files with Wikipedia extracts
- `MODEL_CONFIG` — which LLM handles each role per language
- `MAX_REFINE_LOOPS` — max refinement iterations (default: 2)
- `DRY_RUN` — set True to test without writing output

## Running

```bash
python pipeline_orchestrator.py
```

Output is saved row-by-row to `output/` — safe to interrupt and resume.

## Input Format

Input CSV must have a column named `extract` containing Wikipedia passage text.

## Output Format

| Column | Description |
|--------|-------------|
| `source_text` | Wikipedia passage |
| `language` | English / Italian / Farsi |
| `question_final` | Accepted question |
| `answer_final` | Accepted answer |
| `hint_final` | Accepted Socratic hint |
| `qa_accepted_at_loop` | QA loop index when accepted |
| `hint_accepted_at_loop` | Hint loop index when accepted |
| `recovery_count` | Number of Recovery Agent invocations |
| `qa_q_step_N` / `qa_a_step_N` | Per-loop QA attempts |
| `hint_step_N` | Per-loop hint attempts |
| `*_error_step_N` | Error class per rejection |
