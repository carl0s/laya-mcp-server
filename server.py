"""Laya MCP: an MCP server (plus a small REST API) for Laya typed-decision models.

Laya answers typed questions (choice, score, noul) about a state (text, email,
ticket, JSON) in a single forward pass, with probabilities and confidence.
This server exposes it to MCP clients such as Claude, and to plain HTTP clients
such as n8n.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import os
import secrets
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import uvicorn
from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse

__version__ = "0.1.0"

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("laya-mcp")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _env_list(name: str, default: str = "") -> list[str]:
    return [v.strip() for v in os.getenv(name, default).split(",") if v.strip()]


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
AUTH_MODE = os.getenv("AUTH_MODE", "bearer").strip().lower()  # none | bearer | github
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
CHECKPOINTS = _env_list("LAYA_CHECKPOINTS", "multilingual")
LOW_CONFIDENCE = float(os.getenv("LOW_CONFIDENCE", "0.7"))
LOW_CONFIDENCE_FIELD = os.getenv("LOW_CONFIDENCE_FIELD", "answer_confidence")
MAX_ITEMS = int(os.getenv("MAX_ITEMS", "50"))
API_TOKENS = _env_list("API_TOKENS")
CALIBRATION_PATH = os.getenv("CALIBRATION_PATH", "").strip()

if AUTH_MODE not in {"none", "bearer", "github"}:
    raise SystemExit(f"AUTH_MODE must be none, bearer or github, got '{AUTH_MODE}'")
if AUTH_MODE == "bearer" and not API_TOKENS:
    raise SystemExit("AUTH_MODE=bearer needs at least one token in API_TOKENS")
if AUTH_MODE == "none" and HOST not in {"127.0.0.1", "localhost"}:
    log.warning("AUTH_MODE=none while listening on %s: anyone who can reach this port can use the model", HOST)


# ---------------------------------------------------------------------------
# Calibration (temperature scaling, per question type and option count)
# ---------------------------------------------------------------------------

def _load_calibration() -> dict[str, float]:
    if not CALIBRATION_PATH:
        return {}
    data = json.loads(Path(CALIBRATION_PATH).read_text())
    temps = {str(k): float(v) for k, v in data.get("temperatures", {}).items()}
    log.info("Loaded %d calibration temperatures from %s", len(temps), CALIBRATION_PATH)
    return temps


CALIBRATION = _load_calibration()


def _temperature(qtype: str, n_options: int | None) -> float | None:
    keys = [f"{qtype}:{n_options}"] if n_options else []
    keys += [qtype, "default"]
    for key in keys:
        if key in CALIBRATION:
            return CALIBRATION[key]
    return None


def _calibrate(qtype: str, answer: dict[str, Any]) -> dict[str, Any]:
    """Add a `calibrated` block. Original fields are left untouched."""
    if not CALIBRATION:
        return answer
    if qtype == "choice" and isinstance(answer.get("probabilities"), dict):
        probs = answer["probabilities"]
        t = _temperature("choice", len(probs))
        if t:
            scaled = {k: max(float(p), 1e-12) ** (1.0 / t) for k, p in probs.items()}
            total = sum(scaled.values())
            scaled = {k: round(v / total, 4) for k, v in scaled.items()}
            answer["calibrated"] = {
                "temperature": t,
                "probabilities": scaled,
                "answer_confidence": max(scaled.values()),
            }
    elif qtype == "noul" and isinstance(answer.get("noul"), (int, float)):
        t = _temperature("noul", None)
        if t:
            p = min(max(float(answer["noul"]), 1e-6), 1 - 1e-6)
            q = 1 / (1 + math.exp(-math.log(p / (1 - p)) / t))
            answer["calibrated"] = {"temperature": t, "noul": round(q, 4), "answer_confidence": round(max(q, 1 - q), 4)}
    return answer


def _confidence(answer: dict[str, Any]) -> float | None:
    calibrated = answer.get("calibrated") or {}
    value = calibrated.get("answer_confidence", answer.get(LOW_CONFIDENCE_FIELD))
    return float(value) if isinstance(value, (int, float)) else None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

_router = None
_lock = threading.Lock()  # one forward pass at a time: safer and faster on CPU


def load_model() -> None:
    global _router
    from laya import Router  # imported here so the rest of the module loads without torch

    log.info("Loading Laya checkpoints: %s (first run downloads the weights)", ", ".join(CHECKPOINTS))
    _router = Router(max_loaded=len(CHECKPOINTS))
    _router.preload(CHECKPOINTS)
    log.info("Model ready")


def _validate(questions: dict[str, Any]) -> None:
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object: {name: {type, instructions, criteria}}")
    for name, spec in questions.items():
        qtype = (spec or {}).get("type")
        if qtype not in {"choice", "score", "noul"}:
            raise ValueError(f"Question '{name}': type must be choice, score or noul")
        if qtype == "choice" and not isinstance(spec.get("criteria"), dict):
            raise ValueError(f"Question '{name}': choice needs criteria as an object {{option: description}}")
        if qtype == "score" and not isinstance(spec.get("criteria"), list):
            raise ValueError(f"Question '{name}': score needs criteria as a list of levels")


def _predict(state: dict[str, Any], questions: dict[str, Any], checkpoint: str | None = None) -> dict[str, Any]:
    if _router is None:
        raise RuntimeError("Model is still loading, retry in a moment")
    if checkpoint and checkpoint not in CHECKPOINTS:
        raise ValueError(f"Checkpoint '{checkpoint}' is not loaded. Available: {', '.join(CHECKPOINTS)}")
    model = checkpoint or (CHECKPOINTS[0] if len(CHECKPOINTS) == 1 else None)
    with _lock:
        result = _router.predict(state, questions, model=model) if model else _router.predict(state, questions)
    answers = {q: _calibrate(questions[q]["type"], dict(a)) for q, a in result["answers"].items()}
    return {"answers": answers, "model": (result.get("routing") or {}).get("model", model)}


def classify_one(state: dict, questions: dict, checkpoint: str | None = None) -> dict:
    _validate(questions)
    return _predict(state, questions, checkpoint)


def classify_batch(items: list[dict], questions: dict, checkpoint: str | None = None, include_all: bool = False) -> dict:
    _validate(questions)
    if not items:
        raise ValueError("items is empty")
    if len(items) > MAX_ITEMS:
        raise ValueError(f"At most {MAX_ITEMS} items per call, got {len(items)}. Split the batch.")

    counts = {q: Counter() for q in questions}
    sums = {q: 0.0 for q in questions}
    low, rows = [], []
    for i, state in enumerate(items):
        answers = _predict(state, questions, checkpoint)["answers"]
        item_id = state.get("id") if isinstance(state, dict) else None
        for q, a in answers.items():
            qtype = questions[q]["type"]
            if qtype == "choice":
                counts[q][a.get("choice")] += 1
            elif qtype == "score":
                sums[q] += float(a.get("score", 0))
            else:
                sums[q] += float((a.get("calibrated") or {}).get("noul", a.get("noul", 0)))
            conf = _confidence(a)
            if conf is not None and conf < LOW_CONFIDENCE:
                low.append({"index": i, "id": item_id, "question": q, "answer": a})
        if include_all:
            rows.append({"index": i, "id": item_id, "answers": answers})

    n = len(items)
    summary = {
        q: dict(counts[q]) if spec["type"] == "choice" else {"mean": round(sums[q] / n, 4)}
        for q, spec in questions.items()
    }
    out = {"n": n, "summary": summary, "low_confidence_threshold": LOW_CONFIDENCE, "low_confidence": low}
    if include_all:
        out["items"] = rows
    return out


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _build_oauth():
    """GitHub OAuth, needed for claude.ai custom connectors."""
    from cryptography.fernet import Fernet
    from fastmcp.server.auth.providers.github import GitHubProvider
    from key_value.aio.stores.disk import DiskStore
    from key_value.aio.wrappers.encryption import FernetEncryptionWrapper

    secret = _require("APP_SECRET")
    fernet_key = base64.urlsafe_b64encode(hashlib.sha256(("fernet:" + secret).encode()).digest())
    (DATA_DIR / "oauth").mkdir(parents=True, exist_ok=True)
    return GitHubProvider(
        client_id=_require("GITHUB_CLIENT_ID"),
        client_secret=_require("GITHUB_CLIENT_SECRET"),
        base_url=_require("BASE_URL").rstrip("/"),
        jwt_signing_key=secret,
        # Client registrations and tokens survive restarts, encrypted at rest
        client_storage=FernetEncryptionWrapper(key_value=DiskStore(directory=str(DATA_DIR / "oauth")), fernet=Fernet(fernet_key)),
    )


ALLOWED_GITHUB_USERS = {u.lower() for u in _env_list("ALLOWED_GITHUB_USERS")}
if AUTH_MODE == "github" and not ALLOWED_GITHUB_USERS:
    raise SystemExit("AUTH_MODE=github needs ALLOWED_GITHUB_USERS: GitHub lets any account complete the login")


def _check_github_user() -> None:
    if AUTH_MODE != "github":
        return
    from fastmcp.server.dependencies import get_access_token

    token = get_access_token()
    login = str((token.claims or {}).get("login", "")).lower() if token else ""
    if login not in ALLOWED_GITHUB_USERS:
        raise PermissionError(f"GitHub user '{login or 'unknown'}' is not allowed on this server")


def _token_ok(header: str) -> bool:
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    return bool(token) and any(secrets.compare_digest(token, t) for t in API_TOKENS)


class BearerAuth:
    """Plain ASGI middleware: every path except the open ones needs a valid bearer token."""

    def __init__(self, app, open_paths: set[str]):
        self.app = app
        self.open_paths = open_paths

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] not in self.open_paths:
            headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
            if not _token_ok(headers.get("authorization", "")):
                response = JSONResponse({"error": "unauthorized"}, status_code=401, headers={"WWW-Authenticate": "Bearer"})
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP("laya", auth=_build_oauth() if AUTH_MODE == "github" else None)

TOOL_HINTS = {"readOnlyHint": True, "idempotentHint": True}


@mcp.tool(annotations=TOOL_HINTS)
def laya_classify(state: dict, questions: dict, checkpoint: str | None = None) -> dict:
    """Answer typed questions about one state (text, email, ticket, or any JSON object).

    questions: {"name": {"type": "choice" | "score" | "noul", "instructions": "...", "criteria": ...}}
    - choice: criteria is an object {option: description}; returns choice, probabilities, confidence.
    - score: criteria is a list of ordered levels; returns a score on that scale.
    - noul: yes/no probability that the instruction is true for the state.
    Probabilities from the base model are overconfident: check `calibrated` when present.
    """
    _check_github_user()
    return classify_one(state, questions, checkpoint)


@mcp.tool(annotations=TOOL_HINTS)
def laya_classify_many(items: list[dict], questions: dict, checkpoint: str | None = None, include_all: bool = False) -> dict:
    """Classify up to MAX_ITEMS states with the same questions.

    Returns per-question aggregates (counts for choice, mean for score and noul) and the
    answers below the low-confidence threshold. Set include_all=true to also get every
    answer, which you need when measuring accuracy against labeled data.
    An optional "id" field in each item is echoed back.
    """
    _check_github_user()
    return classify_batch(items, questions, checkpoint, include_all)


@mcp.tool(annotations=TOOL_HINTS)
def laya_info() -> dict:
    """Server configuration: loaded checkpoints, limits, thresholds, calibration."""
    _check_github_user()
    return {
        "version": __version__,
        "checkpoints": CHECKPOINTS,
        "ready": _router is not None,
        "max_items": MAX_ITEMS,
        "low_confidence_threshold": LOW_CONFIDENCE,
        "low_confidence_field": "calibrated.answer_confidence" if CALIBRATION else LOW_CONFIDENCE_FIELD,
        "calibration": CALIBRATION,
    }


# ---------------------------------------------------------------------------
# HTTP routes (health check and REST API for non-MCP clients)
# ---------------------------------------------------------------------------

@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(request: Request) -> JSONResponse:
    ready = _router is not None
    return JSONResponse({"status": "ok" if ready else "loading", "version": __version__}, status_code=200 if ready else 503)


@mcp.custom_route("/v1/classify", methods=["POST"])
async def rest_classify(request: Request) -> JSONResponse:
    """POST {"state": {...}} or {"items": [...]}, plus "questions", optional "checkpoint" and "include_all"."""
    if AUTH_MODE != "none" and not _token_ok(request.headers.get("authorization", "")):
        return JSONResponse({"error": "unauthorized"}, status_code=401, headers={"WWW-Authenticate": "Bearer"})
    try:
        body = await request.json()
        questions = body.get("questions")
        checkpoint = body.get("checkpoint")
        if "items" in body:
            result = classify_batch(body["items"], questions, checkpoint, bool(body.get("include_all")))
        else:
            result = classify_one(body.get("state") or {}, questions, checkpoint)
        return JSONResponse(result)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)


def create_app():
    middleware = [Middleware(BearerAuth, open_paths={"/healthz"})] if AUTH_MODE == "bearer" else []
    # Stateless JSON responses work behind any reverse proxy, including ones that speak HTTP/1.0
    return mcp.http_app(path="/mcp", middleware=middleware, stateless_http=True, json_response=True)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    load_model()
    log.info("Laya MCP %s on http://%s:%d/mcp (auth: %s)", __version__, HOST, PORT, AUTH_MODE)
    uvicorn.run(create_app(), host=HOST, port=PORT, proxy_headers=True, forwarded_allow_ips="*")


if __name__ == "__main__":
    main()
