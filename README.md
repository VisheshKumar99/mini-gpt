# mini-gpt

A from-scratch GPT you can train, watch, and talk to. This repo is a learning
project: a character-level transformer implemented in PyTorch, wrapped in a
FastAPI backend and a React dashboard so you can launch training runs, stream
live loss curves, and chat with a trained model — all locally.

The end goal is to grow this into a small assistant-style model that can hold a
conversation. Today it is a **character-level next-token predictor**: it
continues your text in the style of its training data. The
[roadmap](#roadmap) below is the path from here to something that actually chats.

---

## Table of contents

- [What this is (and isn't) yet](#what-this-is-and-isnt-yet)
- [Architecture](#architecture)
- [Project layout](#project-layout)
- [Getting started](#getting-started)
- [Everyday commands](#everyday-commands)
- [How it works end to end](#how-it-works-end-to-end)
- [Data](#data)
- [Roadmap](#roadmap)
- [What to learn next](#what-to-learn-next)
- [Troubleshooting](#troubleshooting)

---

## What this is (and isn't) yet

**It is:** a working GPT training and inference stack. You define a run
(model size + dataset + hyperparameters), train it, see live metrics, and
generate text from the trained weights.

**It isn't yet:** a chatbot. The models are small (10K–100M+ params on your
hardware), character-level, and trained on plain text with no notion of
"user" and "assistant" turns. Ask it a question and it will not answer — it
will continue the characters you typed in the style of its training corpus.
Turning "text continuation" into "conversation" is what the roadmap covers.

Set expectations honestly: a 10K-parameter model on 1 MB of text overfits into
gibberish. Readable output needs a bigger model, more data, and more steps.

---

## Architecture

```
┌─────────────────┐        HTTP + WebSocket        ┌──────────────────────┐
│  React dashboard │  ─────────────────────────▶   │   FastAPI backend     │
│  (Vite, :5173)   │  ◀─────  live metrics  ─────   │   (uvicorn, :8000)    │
└─────────────────┘                                │                       │
       │                                           │  ┌─────────────────┐  │
       │ Training page: launch + compare runs      │  │ ExperimentRunner│  │
       │ Chat page:     prompt a trained model     │  │  (PyTorch train)│  │
       └───────────────────────────────────────────┼─▶│                 │  │
                                                    │  └────────┬────────┘  │
                                                    │           │ saves     │
                                                    │           ▼           │
                                                    │   models/experiment_  │
                                                    │        {id}.pt         │
                                                    └───────────┬───────────┘
                                                                │ reads
                                                                ▼
                                                        generation service
                                                        (loads checkpoint,
                                                         samples tokens)
```

**The model** (`src/`) is a standard decoder-only transformer built up from
first principles:

- `tokenizer.py` — `CharacterTokenizer`, maps each unique character to an id.
- `embedding.py` — token + learned positional embeddings.
- `attention.py` / `attention_multi.py` — causal self-attention, single and
  multi-head.
- `transformer.py` — a transformer block (attention + feed-forward + norms).
- `gpt.py` — the full `GPT` model stacking N transformer blocks + LM head.
- `loss.py`, `train.py`, `generate.py` — standalone training/generation scripts.

**The backend** (`backend/app/`) turns that into a service:

- `services/experiment_runner.py` — runs a training loop, streams per-step
  metrics, and saves a checkpoint per experiment.
- `services/generation.py` — loads a saved checkpoint and samples text.
- `services/websocket_manager.py` — pushes live metrics to the dashboard.
- `api/experiments.py` — REST + WebSocket endpoints.
- `database.py` — SQLite persistence for experiments and metrics.

**The dashboard** (`dashboard/`) is a Vite + React app with two pages:
a **Training** page (configure, launch, and compare runs with live charts) and
a **Chat** page (pick a trained model and prompt it).

---

## Project layout

```
mini-gpt/
├── src/                       # the model, from scratch (PyTorch)
│   ├── tokenizer.py           # character-level tokenizer
│   ├── embedding.py           # token + position embeddings
│   ├── attention.py           # single-head causal attention
│   ├── attention_multi.py     # multi-head attention
│   ├── transformer.py         # one transformer block
│   ├── gpt.py                 # the GPT model
│   ├── train.py               # standalone training script
│   ├── generate.py            # standalone generation script
│   └── test/                  # unit tests for the components
├── backend/
│   └── app/
│       ├── main.py            # FastAPI app + WebSocket route
│       ├── database.py        # SQLite (experiments, metrics)
│       ├── api/experiments.py # REST + WS endpoints
│       ├── models/            # Pydantic schemas
│       └── services/          # runner, generation, metrics, ws manager
├── dashboard/                 # React + Vite frontend
│   └── src/
│       ├── pages/Dashboard.jsx  # training + comparison
│       ├── pages/Chat.jsx       # chat with a trained model
│       ├── api/Experiments.js   # backend client
│       └── components/          # form, charts, cards
├── data/                      # training corpora (git-ignored above 1 MB)
├── models/                    # saved checkpoints (git-ignored)
├── start.sh                   # run backend + frontend together
└── requirements files, .gitignore, etc.
```

---

## Getting started

**Prerequisites:** Python 3.10+ (the code uses 3.10 type syntax), Node 18+,
and [Homebrew](https://brew.sh) on macOS for installing them.

```bash
# 1. Python 3.10 (if you don't have it)
brew install python@3.10

# 2. Create and activate a virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# 3. Install backend dependencies
pip install -r backend/requirements.txt

# 4. Install frontend dependencies
npm install --prefix dashboard
```

Then start both servers together:

```bash
./start.sh
```

- Backend API: http://localhost:8000 (docs at http://localhost:8000/docs)
- Dashboard:   http://localhost:5173

To use the app: open the dashboard, configure a run on the **Training** page and
launch it, wait for it to finish, then switch to the **Chat** page, pick that
run, and prompt it.

---

## Everyday commands

```bash
# Activate the environment (needed in every new terminal)
source .venv/bin/activate

# Run backend + frontend together
./start.sh

# Run only the backend
uvicorn backend.app.main:app --reload --port 8000

# Run only the frontend
npm run dev --prefix dashboard

# Build the frontend for production
npm run build --prefix dashboard

# Run the model unit tests
python -m pytest src/test

# Generate a training corpus (streams TinyStories)
pip install datasets            # one-time
python data/download_data.py 100        # -> data/100mb.txt
python data/download_data.py 10         # -> data/10mb.txt

# Inspect the database
sqlite3 experiments.db
#   .tables
#   SELECT id, name, status FROM experiments;
#   DELETE FROM experiments WHERE status='running';
#   .quit

# Free a port that's stuck "Address already in use"
lsof -ti:8000 | xargs kill        # or kill -9 to force
```

---

## How it works end to end

1. **Create an experiment** — the dashboard posts a config (model size,
   dataset, batch size, learning rate, steps) to `POST /api/experiments`.
2. **Train** — `POST /api/experiments/{id}/train` runs `ExperimentRunner` in a
   background task. Each step it computes train/val loss and throughput, writes
   the metric to the `experiment_metrics` table, and publishes it over the
   WebSocket so the chart updates live.
3. **Persist** — when training finishes, the runner saves
   `models/experiment_{id}.pt` (weights + architecture + tokenizer vocabulary),
   and a summary is stored on the experiment row.
4. **Chat** — the Chat page lists runs where `has_model` is true. Sending a
   prompt hits `POST /api/experiments/{id}/generate`, which loads the checkpoint
   and samples a continuation with temperature + top-k sampling.

### Key API endpoints

| Method | Path                                | Purpose                          |
|--------|-------------------------------------|----------------------------------|
| GET    | `/api/experiments`                  | list runs (incl. `has_model`)    |
| POST   | `/api/experiments`                  | create a run                     |
| POST   | `/api/experiments/{id}/train`       | start training (background)      |
| GET    | `/api/experiments/{id}/metrics`     | per-step metric history          |
| POST   | `/api/experiments/{id}/generate`    | sample text from a trained model |
| WS     | `/ws/experiments/{id}`              | live metric stream               |

---

## Data

Training corpora live in `data/`. The dashboard catalog expects files named
`data/1mb.txt`, `data/10mb.txt`, `data/100mb.txt`, `data/1gb.txt`. Files larger
than 1 MB are git-ignored (regenerate them with the download script). The
default source is [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories),
a corpus of simple synthetic children's stories — good for small models because
the vocabulary and grammar are limited.

---

## Roadmap

A 10-stage path from "continues text" to "holds a conversation." Stages 1–5 are
largely done; 6 onward is the learning journey toward a chat model.

### Stage 1 — Foundations (done)
Build the transformer from scratch: tokenizer, embeddings, attention,
transformer block, GPT model. Understand each tensor shape as data flows through.

### Stage 2 — Training loop (done)
Batching, cross-entropy loss, AdamW, train/val split, and tracking loss over
steps. Learn to read a loss curve and spot overfitting (val loss rising while
train loss falls).

### Stage 3 — Serving + observability (done)
Wrap training in a FastAPI service, persist experiments in SQLite, and stream
live metrics to a dashboard over WebSockets. This makes experiments repeatable
and comparable.

### Stage 4 — Checkpointing + inference (done)
Save trained weights per experiment and load them for sampling with temperature
and top-k. This is the difference between "a training script" and "a model you
can reuse."

### Stage 5 — Chat surface (done)
A UI to select a trained model and prompt it. Right now it does raw text
continuation — the honest current capability.

### Stage 6 — Better tokenization
Move from character-level to **subword** tokenization (BPE, e.g. `tiktoken` or
a trained `sentence-piece`/`tokenizers` model). Subwords shorten sequences,
enlarge effective context, and dramatically improve sample quality. This is the
single biggest quality jump available.

### Stage 7 — Scale the model and data
Train larger configs (more layers/heads/embedding width) on more data (100 MB→1 GB),
with a proper learning-rate schedule (warmup + cosine decay), gradient clipping,
and checkpoints saved periodically (not just at the end). Learn what actually
moves validation loss.

### Stage 8 — Instruction / chat formatting
Introduce a **chat template**: structure data as `system` / `user` / `assistant`
turns with special delimiter tokens. Fine-tune on instruction–response pairs so
the model learns to answer rather than continue. This is "supervised
fine-tuning" (SFT) and is where it starts to feel like a chatbot.

### Stage 9 — Alignment and quality
Explore preference tuning (DPO or RLHF-lite) on ranked responses, plus
guardrails: stop sequences, max-length handling, repetition penalties, and
refusal behavior. Learn to evaluate quality beyond loss (held-out prompts,
human or LLM-judged comparisons).

### Stage 10 — Product hardening
Streaming token-by-token responses over WebSocket, multi-turn conversation
memory in the UI, per-conversation context windows, request queuing, model
versioning, and basic auth. Turn the demo into something other people can use.

---

## What to learn next

Concretely, to fine-tune this and get better responses, study these in order:

1. **Subword tokenization (BPE).** Read the tokens as the model sees them.
   Libraries: `tiktoken`, HuggingFace `tokenizers`. *Biggest single win.*
2. **Learning-rate schedules.** Warmup + cosine decay; why a flat LR plateaus.
3. **Regularization + scale.** Dropout, weight decay, and how loss responds to
   more parameters vs. more data (the scaling-laws intuition).
4. **Sampling strategies.** Temperature, top-k, top-p (nucleus), and repetition
   penalty — how each changes the "voice" of the output.
5. **Chat data formatting.** Special tokens, chat templates, masking the loss so
   the model is only graded on the assistant's reply.
6. **Supervised fine-tuning (SFT).** Take a pretrained base and fine-tune on
   instruction–response pairs. Start from a small open base model if training
   from scratch is too slow locally.
7. **Evaluation.** Perplexity is not quality. Build a fixed prompt suite and
   compare runs side by side.
8. **Efficient fine-tuning.** LoRA / QLoRA to fine-tune larger models on a
   single machine.

A realistic local target: subword tokenizer + a few-million-parameter model on
100 MB+ of text, fine-tuned on a small instruction dataset, with token streaming
in the UI. That combination is what makes it feel like a real assistant.

---

## Troubleshooting

**`command not found: pip`** — activate the venv first (`source .venv/bin/activate`),
or use `python3 -m pip`.

**`TypeError: unsupported operand type(s) for |`** — you're on Python < 3.10.
Recreate the venv with `python3.10 -m venv .venv`.

**`ModuleNotFoundError: No module named 'torch'` (or `numpy`)** — install deps:
`pip install -r backend/requirements.txt`.

**`[Errno 48] Address already in use`** — a previous server still holds the port.
Free it: `lsof -ti:8000 | xargs kill`.

**`[Errno 9] Bad file descriptor` during data download** — harmless. It's the
streaming client shutting down after your file was already written. Check the
file with `ls -lh data/100mb.txt`.

**Chat page says "No trained models yet"** — you need a run whose status is
`completed`, which writes a checkpoint to `models/`. Train one first, then
refresh.
