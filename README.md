# AuroraStay policy desk

An internal-facing hotel-policy RAG proof of concept with a Flask API and
Streamlit interface. It parses the four supplied AuroraStay policy PDFs with
Docling, stores 133 source-cited chunks in native FAISS plus a JSON docstore,
and uses a pinned hotel embedding checkpoint. Dense and BM25 retrieval are
combined with property identity and duplicate-row handling.

Generation uses gpt-4o-mini for a draft and grounding edit. Material policy
conditions are copied from the retrieved evidence. The app displays a staff
review notice and has no booking, refund, payment or messaging tools.

## Local setup

Use Python 3.12. Install the tested direct requirements in a virtual environment:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create a private `config.json` using `config.example.json`, or set
`OPENAI_API_KEY` and `OPENAI_API_BASE` in the process environment. The private
configuration is ignored by Git and excluded from Docker build contexts.
Never commit or publish it.

Run the two services in separate terminals from the project root:

```sh
python deployment/backend/app.py
```

```sh
streamlit run deployment/frontend/app.py --server.port 8501 --browser.gatherUsageStats false
```

The local backend is at `http://127.0.0.1:7860`; the UI is at
`http://127.0.0.1:8501`. The embedding checkpoint downloads on its first load.
The index manifest validates dimensions, hashes, checkpoint and retrieval
configuration before loading. No pickle is deserialized.

## Docker and Codespaces

The backend and frontend Dockerfiles use Python 3.12. The backend installs CPU
PyTorch and caches the exact embedding revision during the build. From the
project root, with the two OpenAI environment variables set:

```sh
docker compose -f deployment/docker-compose.yml up --build -d
```

The devcontainer configuration retains GitHub's default development image and
forwards 7860/8501. Dependencies live in the two application images; there is
no duplicate host-level GPU/PyTorch installation. Keep forwarded ports private
except for an explicitly approved temporary demonstration. The UI uses the
backend service name on the Compose network.

The course's Codespaces guide describes port forwarding, screenshots and
cleanup. Stop a temporary Codespace after capturing evidence and delete it
when its saved work is no longer needed. Review GitHub's usage page before
starting resources; this project does not enable billing or purchase services.

## HTTP interface

- `GET /health`
- `POST /v1/relevant_chunks`
- `POST /v1/answer`
- `POST /v1/answer_with_relevant_chunks`

Requests contain `query` and `k`; answer requests can also specify `model_name`,
`temperature`, `top_p` and `max_tokens`. The default is k=4, gpt-4o-mini,
temperature=0, top_p=1 and max_tokens=512. The model's embedding tokenizer
has a 128-token question limit. Invalid inputs, unavailable indexes, upstream
failures and deadlines produce bounded error responses without credentials.
An HTTP deadline limits client waiting; an already-running provider request
may finish afterward.

## Evaluation

The supplied evaluation and test CSVs are separate from the retrieval corpus.
DeepEval G-Eval uses an independent gpt-4o judge for four metrics. The weighted
composite is 0.35 faithfulness + 0.25 contextual recall + 0.20 contextual
precision + 0.20 answer relevancy. The benchmark AUTO boundary is inclusive
at 0.75. Review/manual cases count as non-auto business workload.

Contextual recall requires a reference answer, so this offline composite is
not a live confidence score. Benchmark AUTO eligibility is not measured hotel
query deflection. All live outputs remain staff-review drafts.

The course handoff contains the executed HTML notebook, final benchmark CSVs,
JSON traces, source hashes, a frozen configuration, development audits and
actual deployment evidence. Development revisions and an early unreviewed
test launch are retained separately, rather than mixed into final metrics.

## Checks

```sh
python -m unittest deployment.test_http
```

The project retains native-index round-trip, token-length, property-row,
concurrent-query, input-contract and UI checks. Docker/cloud verification
must be supported by the actual deployment evidence, not inferred from these
local checks.

Source references: [pinned embedding](https://huggingface.co/HarishMaths/Hotel-Policy-Embedding/tree/3c9f67da8057fb1e9f742c9e6680d5a4b33b4dcd),
[DeepEval G-Eval](https://deepeval.com/docs/metrics-llm-evals),
[GitHub default dev containers](https://docs.github.com/en/codespaces/setting-up-your-project-for-codespaces/adding-a-dev-container-configuration/introduction-to-dev-containers),
[port forwarding](https://docs.github.com/en/codespaces/developing-in-a-codespace/forwarding-ports-in-your-codespace).
