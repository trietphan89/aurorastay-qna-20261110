# AuroraStay application

The Flask API and Streamlit UI use the saved native FAISS index and JSON docstore in `artifacts/faiss_index`. They use the same `rag_core.py` and `retrieval.py` as the notebook. No pickle file is loaded, and the inference image does not need Docling or the benchmark CSVs. The inference requirements preserve the tested Transformers5.17.0, Hugging Face Hub1.16.0 and Click8.3.3 compatibility constraints. CPU threading defaults to one thread per numerical library to avoid native threading conflicts.

## Run locally

From the project root, install the pinned packages:

```sh
python -m pip install -r deployment/backend/requirements.txt -r deployment/frontend/requirements.txt
```

Set `OPENAI_API_KEY` and `OPENAI_API_BASE` in the terminal environment or an approved secret manager. The API does not accept credentials in request bodies, and no credentials belong in the repository. The project `.streamlit/config.toml` provides the light hotel theme and headless startup. Then start these in two terminals:

```sh
python deployment/backend/app.py
```

```sh
streamlit run deployment/frontend/app.py --server.port 8501 --server.address 127.0.0.1
```

Open `http://127.0.0.1:8501`. The backend listens on `127.0.0.1:7860` by default. If the index cannot load, `/health` returns503; repair the artifact/configuration and restart the server. Set `INDEX_DIR` to use a different saved index location. `RAG_REQUEST_TIMEOUT_SEC` defaults90 seconds and is limited to120. Timed-out provider calls may still complete in the bounded worker pool; the application does not claim to cancel work already accepted by the provider.

## Docker

Both Docker builds use the **project root** as build context. The backend caches the pinned embedding checkpoint at build time and loads it offline at runtime.

```sh
docker compose -f deployment/docker-compose.yml up --build
```

Compose requires both OpenAI environment variables and binds ports to localhost. It passes no API key into the build. XSRF protection and Streamlit's default origin protections remain enabled. Stop local containers with `docker compose -f deployment/docker-compose.yml down`.

## Codespaces preparation

The supplied `.devcontainer/devcontainer.json` installs the same dependencies and labels ports7860/8501. It does not create a repository or Codespace, publish a port, or start a billed service. Add the two OpenAI values through GitHub Codespaces secrets and grant the intended repository access, following the supplied course guide. GitHub forwards ports privately by default; keep the staff POC private unless an approved deployment process provides appropriate access controls. Start the two commands above inside a Codespace and obtain the real forwarded URL from the Ports tab. Do not replace it with a fabricated URL in the assignment. Official references: [devcontainer properties](https://github.com/devcontainers/spec/blob/main/docs/specs/devcontainerjson-reference.md), [GitHub port forwarding](https://docs.github.com/en/codespaces/developing-in-a-codespace/forwarding-ports-in-your-codespace).

## API

| Method | Endpoint | Result |
|---|---|---|
| GET | `/health` | Index readiness |
| POST | `/v1/relevant_chunks` | Policy passages |
| POST | `/v1/answer` | Answer and source metadata |
| POST | `/v1/answer_with_relevant_chunks` | Answer, source metadata and full policy passages |

Retrieval payload: `{"query":"What is the cancellation policy?","k":3}`. Generation also accepts `model_name`, `temperature`, `top_p`, `max_tokens`. Query length1–2000 characters, k1–10, temperature0–1, top_p0–1 and max_tokens64–2048 are enforced. Only `gpt-4o-mini` and `gpt-4o` are allowed. Invalid requests return400 before retrieval or model generation; oversized bodies return413, busy/not-ready services503, upstream failures502 and deadline failures504. Error responses include a request ID and do not expose provider messages.

All answers require staff review and contain source metadata. The runtime does **not** present a benchmark composite score: contextual recall needs a reference answer, which arbitrary live questions do not have. The application does not change reservations, issue refunds or contact guests.

## Checks

```sh
python deployment/test_http.py --report build/deployment/contract_checks.json
python deployment/test_http.py --base-url http://127.0.0.1:7860 --report build/deployment/http_checks.json
```

The default contract checks start a real local HTTP server with deterministic retrieval/generation doubles and consume no API tokens. The second command also checks the actual saved index over HTTP. Add `--include-generation` only when a real model call is intended; this checks both answer endpoints and records their actual responses. Keep simulated checks separate from live-model results. Docker and Codespaces deployment status must be reported separately from local execution.
