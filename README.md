# Research Agent

A small research agent built with the Anthropic Python SDK. You give it a
question. It plans, you edit the plan, then it searches the web (Tavily),
optionally searches your private documents, and writes a briefing with
citations in your chosen style.

- `hello.py` - smallest possible API check.
- `agent.py` - the agent loop, planner, and citation formatter. Runnable as a CLI.
- `app.py` - Streamlit UI: input -> editable plan -> streaming run -> briefing + citations.

## Features

- **Planning step.** Claude writes a research plan first. You edit it before
  the agent executes.
- **Adversarial mode.** Toggle to flip the agent into "find counterevidence"
  mode instead of summary mode.
- **Private corpus.** Upload PDFs or text files. The agent searches them
  alongside the web via a `corpus_search` tool.
- **Citation formatting.** Pick APA, MLA, Chicago, IEEE, Harvard, or BibTeX.
  A post-research Claude call formats the sources for you.

## One-time local setup

```bash
cd "/Users/BigAl/Agent V1"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
open -e .env   # paste both keys, save, close
```

You need two keys in `.env`:

- `ANTHROPIC_API_KEY` - https://console.anthropic.com/settings/keys
- `TAVILY_API_KEY` - https://app.tavily.com/ (free tier: 1000 searches/month)

## Run it

```bash
source .venv/bin/activate

# Sanity check the Anthropic key
python3 hello.py

# CLI agent (uses planning + citations end-to-end)
python3 agent.py "What is retrieval-augmented generation?"

# Streamlit UI
streamlit run app.py
```

The Streamlit app opens at http://localhost:8501. Leave the terminal
running - closing it stops the app.

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repo. `.gitignore` already excludes
   `.env` and `.venv`, so your keys will not be committed.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, point it at the repo, and set:
   - **Main file path:** `app.py`
   - **Python version:** 3.11 or higher
4. Open **Advanced settings -> Secrets** and paste:

   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   TAVILY_API_KEY = "tvly-..."
   ```

5. Click **Deploy**. First build takes ~2 minutes.

The app reads both keys from `st.secrets` when deployed and falls back to
your local `.env` on your laptop.

## Troubleshooting

- **`localhost refused to connect`** - the `streamlit run` process isn't
  running. Open Terminal, re-run `streamlit run app.py`, leave it open.
- **`python3: command not found`** - install Python from python.org.
- **`No module named 'anthropic'`** - venv isn't active; re-run
  `source .venv/bin/activate`.
- **`AuthenticationError`** - the Anthropic key is wrong, expired, or
  out of credit.
- **`KeyError: 'TAVILY_API_KEY'`** - your `.env` is missing the Tavily
  key (or you're deployed and haven't set it in Secrets).
- **PDF upload returns garbled text** - `pypdf` can't extract from
  image-only or heavily formatted PDFs. Try a plain text export.
- **Streamlit Cloud build fails** - `requirements.txt` must be at the
  repo root; main file path must be `app.py`.
