"""Bounded REST interface for the AuroraStay document-grounded assistant.

The Flask factory loads the immutable FAISS index once per worker. Generation
and retrieval implementations are injectable for isolated HTTP contract tests.
No booking, refund, notification or other write operation is exposed.
"""
from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import logging
import math
import os
from pathlib import Path
import sys
import threading
import time
import uuid

# Establish CPU thread limits before the lazily imported embedding stack starts.
for _thread_setting in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thread_setting, "1")

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOGGER = logging.getLogger(__name__)
GENERATION_FIELDS = {"query", "k", "model_name", "temperature", "top_p", "max_tokens"}
MODELS = {"gpt-4o-mini", "gpt-4o"}
REVIEW_STATUS = "STAFF_REVIEW_REQUIRED"


class InvalidRequest(ValueError):
    pass


def _integer(data, name, default, lower, upper):
    value = data.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        raise InvalidRequest(f"{name} must be an integer from {lower} to {upper}.")
    return value


def _number(data, name, default, lower, upper):
    value = data.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise InvalidRequest(f"{name} must be a number from {lower} to {upper}.")
    if not math.isfinite(value) or not lower <= value <= upper:
        raise InvalidRequest(f"{name} must be a number from {lower} to {upper}.")
    return float(value)


def _payload(generation=False):
    if not request.is_json:
        raise InvalidRequest("Send a JSON object with Content-Type application/json.")
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise InvalidRequest("The request body must be a valid JSON object.")
    allowed = GENERATION_FIELDS if generation else {"query", "k"}
    if set(data) - allowed:
        raise InvalidRequest("The request contains unsupported fields.")
    query = data.get("query")
    if not isinstance(query, str) or not query.strip():
        raise InvalidRequest("query must be a nonempty string.")
    if len(query) > 2000:
        raise InvalidRequest("query must contain at most 2000 characters.")
    parsed = {"query": query.strip(), "k": _integer(data, "k", 4, 1, 10)}
    if generation:
        model = data.get("model_name", "gpt-4o-mini")
        if not isinstance(model, str) or model not in MODELS:
            raise InvalidRequest("model_name must be gpt-4o-mini or gpt-4o.")
        parsed.update(
            model_name=model,
            temperature=_number(data, "temperature", 0.0, 0.0, 1.0),
            top_p=_number(data, "top_p", 1.0, 0.0, 1.0),
            max_tokens=_integer(data, "max_tokens", 512, 64, 2048),
        )
    return parsed


def create_app(*, vectorstore=None, retriever=None, answerer=None,
               index_dir=None, request_timeout=None):
    """Create one server worker; passing test doubles never reads credentials."""
    app = Flask("AuroraStay Policy API")
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
    timeout = float(request_timeout if request_timeout is not None
                    else os.environ.get("RAG_REQUEST_TIMEOUT_SEC", "90"))
    if not math.isfinite(timeout) or not 0 < timeout <= 120:
        raise ValueError("RAG_REQUEST_TIMEOUT_SEC must be greater than 0 and at most 120.")
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="aurorastay-rag")
    slots = threading.BoundedSemaphore(2)
    app.extensions["rag_worker_pool"] = pool
    atexit.register(pool.shutdown, wait=False, cancel_futures=True)
    startup_error = None
    if retriever is None:
        from retrieval import retrieve
        retriever = retrieve
    if answerer is None:
        from rag_core import answer_query
        answerer = answer_query
    if vectorstore is None:
        try:
            from retrieval import load_index
            vectorstore = load_index(index_dir or os.environ.get(
                "INDEX_DIR", str(ROOT / "artifacts/faiss_index")))
        except Exception as exc:
            startup_error = type(exc).__name__
            LOGGER.error("Index startup failed (%s); restart after repairing the index.", startup_error)

    def error(code, message, status):
        return jsonify(error={"code": code, "message": message},
                       request_id=getattr(request, "rag_request_id", None)), status

    @app.before_request
    def request_id():
        request.rag_request_id = uuid.uuid4().hex

    @app.after_request
    def response_headers(response):
        response.headers["X-Request-ID"] = getattr(request, "rag_request_id", "")
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.errorhandler(InvalidRequest)
    def invalid_request(exc):
        return error("INVALID_REQUEST", str(exc), 400)

    @app.errorhandler(HTTPException)
    def http_error(exc):
        messages = {404: "Endpoint not found.", 405: "Method not allowed.",
                    413: "The request body is too large."}
        return error(f"HTTP_{exc.code}", messages.get(exc.code, "The HTTP request could not be processed."), exc.code)

    def run_operation(operation, params):
        if startup_error or vectorstore is None:
            return None, error("INDEX_UNAVAILABLE", "The policy library is not ready. Contact the application administrator.", 503)
        if not slots.acquire(blocking=False):
            return None, error("SERVER_BUSY", "The assistant is busy. Please try again shortly.", 503)
        started = time.monotonic()
        future = pool.submit(operation, **params, vectorstore=vectorstore)
        future.add_done_callback(lambda _: slots.release())
        try:
            return (future.result(timeout=timeout), round(time.monotonic() - started, 3)), None
        except FutureTimeout:
            future.cancel()
            return None, error("REQUEST_TIMEOUT", "The request exceeded its time limit. Please try again or use manual lookup.", 504)
        except Exception as exc:
            # Provider exception strings can contain request details; never return or log them.
            LOGGER.error("RAG request failed (%s), request_id=%s", type(exc).__name__, request.rag_request_id)
            return None, error("UPSTREAM_ERROR", "The assistant could not complete the request. Please use manual lookup or try again.", 502)

    @app.get("/")
    def home():
        return jsonify(service="AuroraStay Policy API", health="/health")

    @app.get("/health")
    def health():
        ready = vectorstore is not None and startup_error is None
        return jsonify(status="ready" if ready else "unavailable", index_loaded=ready,
                       review_status=REVIEW_STATUS), 200 if ready else 503

    @app.post("/v1/relevant_chunks")
    def relevant_chunks():
        params = _payload()
        output, failure = run_operation(retriever, params)
        if failure:
            return failure
        chunks, elapsed = output
        if not isinstance(chunks, list) or not all(isinstance(c, str) for c in chunks):
            return error("INVALID_UPSTREAM_RESPONSE", "The policy library returned an incomplete response. Please use manual lookup.", 502)
        return jsonify(**params, relevant_chunks=chunks, elapsed_seconds=elapsed,
                       review_status=REVIEW_STATUS)

    def generate(include_chunks):
        params = _payload(generation=True)
        output, failure = run_operation(answerer, params)
        if failure:
            return failure
        result, elapsed = output
        if (not isinstance(result, dict) or not isinstance(result.get("answer"), str)
                or not result.get("answer", "").strip()
                or not isinstance(result.get("relevant_chunks"), list)
                or not all(isinstance(c, str) for c in result.get("relevant_chunks", []))
                or not isinstance(result.get("sources"), list)
                or not all(isinstance(c, dict) for c in result.get("sources", []))):
            return error("INVALID_UPSTREAM_RESPONSE", "The assistant returned an incomplete response. Please use manual lookup.", 502)
        response = dict(params, answer=result["answer"], sources=result["sources"],
                        review_status=REVIEW_STATUS, elapsed_seconds=elapsed)
        if include_chunks:
            response["relevant_chunks"] = result["relevant_chunks"]
        return jsonify(response)

    @app.post("/v1/answer")
    def answer():
        return generate(False)

    @app.post("/v1/answer_with_relevant_chunks")
    def answer_with_relevant_chunks():
        return generate(True)

    return app


if __name__ == "__main__":
    create_app().run(host=os.environ.get("HOST", "127.0.0.1"),
                     port=int(os.environ.get("PORT", "7860")), debug=False)
