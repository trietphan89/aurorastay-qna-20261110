"""HTTP contract checks, plus optional checks against the actual local server.

Default tests use an in-process HTTP server with deterministic test doubles:
they verify request boundaries without consuming API tokens. The --base-url
option separately checks the real policy index via HTTP. Generation on the
real server is opt-in via --include-generation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import threading
import time
import unittest

import requests
from werkzeug.serving import make_server

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("aurorastay_backend", ROOT / "deployment/backend/app.py")
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


class HTTPContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calls = {"retrieve": 0, "answer": 0}

        def retrieve(query, k, vectorstore):
            cls.calls["retrieve"] += 1
            return [f"Policy passage {i+1}: {query}" for i in range(k)]

        def answer_query(query, k, model_name, temperature, top_p, max_tokens, vectorstore):
            cls.calls["answer"] += 1
            if query == "simulate provider timeout":
                time.sleep(.15)
            if query == "simulate provider error":
                raise RuntimeError("PRIVATE_PROVIDER_DETAILS_MUST_NOT_LEAK")
            return {"answer": "Policy-grounded test draft.",
                    "relevant_chunks": [f"Policy passage {i+1}" for i in range(k)],
                    "sources": [{"source": "policy.pdf", "page": 2, "chunk_id": "policy-p2-c1"}]}

        cls.app = backend.create_app(vectorstore=object(), retriever=retrieve,
                                     answerer=answer_query, request_timeout=.05)
        cls.server = make_server("127.0.0.1", 0, cls.app, threaded=True)
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        cls.app.extensions["rag_worker_pool"].shutdown(wait=True)

    def post(self, path, data):
        return requests.post(self.url + path, json=data, timeout=2)

    def test_health_and_no_store(self):
        response = requests.get(self.url + "/health", timeout=2)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["index_loaded"])
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertTrue(response.headers["X-Request-ID"])

    def test_invalid_inputs_do_not_invoke_rag(self):
        before = dict(self.calls)
        invalid = [None, [], {}, {"query": " "}, {"query": 5}, {"query": "x" * 2001},
                   {"query": "parking", "k": 0}, {"query": "parking", "k": 11},
                   {"query": "parking", "k": True}, {"query": "parking", "k": 2.5},
                   {"query": "parking", "temperature": -1},
                   {"query": "parking", "temperature": 1.1},
                   {"query": "parking", "temperature": True},
                   {"query": "parking", "top_p": 1.1},
                   {"query": "parking", "max_tokens": 63},
                   {"query": "parking", "max_tokens": 2049},
                   {"query": "parking", "model_name": "unapproved-model"},
                   {"query": "parking", "api_key": "never-accept-request-secrets"}]
        for payload in invalid:
            with self.subTest(payload=payload):
                response = self.post("/v1/answer", payload)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()["error"]["code"], "INVALID_REQUEST")
        self.assertEqual(self.calls, before)

    def test_malformed_non_json_and_large_body(self):
        response = requests.post(self.url + "/v1/answer", data="not-json", timeout=2)
        self.assertEqual(response.status_code, 400)
        response = requests.post(self.url + "/v1/answer", data="{", headers={"Content-Type": "application/json"}, timeout=2)
        self.assertEqual(response.status_code, 400)
        response = requests.post(self.url + "/v1/answer", data="x" * 17000,
                                 headers={"Content-Type": "application/json"}, timeout=2)
        self.assertEqual(response.status_code, 413)

    def test_retrieval_and_generation_shapes(self):
        response = self.post("/v1/relevant_chunks", {"query": "parking", "k": 2})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["relevant_chunks"]), 2)
        answer = self.post("/v1/answer", {"query": "parking", "k": 2})
        self.assertEqual(answer.status_code, 200)
        self.assertIn("answer", answer.json())
        self.assertNotIn("relevant_chunks", answer.json())
        full = self.post("/v1/answer_with_relevant_chunks", {"query": "parking", "k": 2})
        self.assertEqual(full.status_code, 200)
        self.assertEqual(len(full.json()["relevant_chunks"]), 2)
        self.assertEqual(full.json()["sources"][0]["page"], 2)
        self.assertEqual(full.json()["review_status"], "STAFF_REVIEW_REQUIRED")
        self.assertNotIn("composite_score", full.json())

    def test_parameter_boundaries_are_valid(self):
        for params in [{"k": 1, "temperature": 0, "top_p": 0, "max_tokens": 64},
                       {"k": 10, "temperature": 1, "top_p": 1, "max_tokens": 2048}]:
            response = self.post("/v1/answer", dict(query="policy", **params))
            self.assertEqual(response.status_code, 200)

    def test_upstream_error_is_redacted(self):
        response = self.post("/v1/answer", {"query": "simulate provider error"})
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("PRIVATE_PROVIDER", response.text)

    def test_timeout_is_bounded(self):
        start = time.monotonic()
        response = self.post("/v1/answer", {"query": "simulate provider timeout"})
        self.assertEqual(response.status_code, 504)
        self.assertLess(time.monotonic() - start, .14)
        time.sleep(.16)  # Let the bounded test-double worker finish before later cases.


def real_server_checks(base_url, include_generation=False):
    base_url = base_url.rstrip("/")
    report = {"base_url": base_url, "actual_index_http": True,
              "live_generation_requested": include_generation, "checks": []}
    response = requests.get(base_url + "/health", timeout=(5, 120))
    response.raise_for_status()
    assert response.json().get("index_loaded") is True
    report["checks"].append({"endpoint": "/health", "status": response.status_code})
    payload = {"query": "For ASH-SEA Pike, is breakfast included and what should I budget for parking?", "k": 3}
    response = requests.post(base_url + "/v1/relevant_chunks", json=payload, timeout=(5, 95))
    response.raise_for_status()
    result = response.json()
    assert len(result["relevant_chunks"]) == 3
    assert all(isinstance(c, str) and len(c) > 30 for c in result["relevant_chunks"])
    report["checks"].append({"endpoint": "/v1/relevant_chunks", "status": response.status_code,
                             "payload": payload, "response": result})
    invalid = requests.post(base_url + "/v1/answer", json={"query": "parking", "k": 0}, timeout=(5, 15))
    assert invalid.status_code == 400
    report["checks"].append({"endpoint": "/v1/answer", "invalid_k_status": invalid.status_code})
    if include_generation:
        payload.update(model_name="gpt-4o-mini", temperature=0, top_p=1, max_tokens=512)
        for endpoint in ["/v1/answer", "/v1/answer_with_relevant_chunks"]:
            response = requests.post(base_url + endpoint, json=payload, timeout=(5, 95))
            response.raise_for_status()
            result = response.json()
            assert isinstance(result["answer"], str) and result["answer"].strip()
            assert result["sources"]
            if endpoint.endswith("chunks"):
                assert len(result["relevant_chunks"]) == payload["k"]
            report["checks"].append({"endpoint": endpoint, "status": response.status_code,
                                     "payload": dict(payload), "response": result})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", help="Also verify an actual running backend; retrieval only by default")
    parser.add_argument("--include-generation", action="store_true", help="Consume API calls for both answer endpoints")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(HTTPContractTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"contract_test_cases": result.testsRun, "failures": len(result.failures),
              "errors": len(result.errors), "passed": result.wasSuccessful()}
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.base_url:
        report["server"] = real_server_checks(args.base_url, args.include_generation)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "server"}, indent=2))
