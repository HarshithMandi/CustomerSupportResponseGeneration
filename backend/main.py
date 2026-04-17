from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .bm25_retriever import BM25PolicyIndex
from .logging_utils import log_event, setup_logger
from .prompts import build_prompt, fallback_response
from .sarvam_client import SarvamLLM, strip_think_tags


DATA_PATH = os.getenv("POLICY_DATA_PATH") or os.path.join(os.path.dirname(__file__), "data", "policies.json")
XLSX_PATH = os.getenv("XLSX_DATA_PATH") or os.path.join(os.path.dirname(__file__), "data", "Complaint Dataset.xlsx")
BM25_MIN_SCORE = float(os.getenv("BM25_MIN_SCORE") or "0.5")

logger = setup_logger(os.path.join(os.path.dirname(__file__), "logs"))

xlsx_file: str | None = None
if Path(XLSX_PATH).exists():
    xlsx_file = XLSX_PATH

# Build separate indexes so we can reliably include *policies* as well as the XLSX dataset.
policy_index = BM25PolicyIndex.from_json_file(DATA_PATH)
playbook_index = BM25PolicyIndex.from_sources(json_path=None, xlsx_path=xlsx_file)

app = FastAPI(title="AI-CSRG Backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    mode: str = Field(default="strict")  # strict | friendly
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=1000)


class RetrievedDocOut(BaseModel):
    title: str
    content: str
    score: float


class GenerateResponse(BaseModel):
    response: str
    retrieved_docs: list[RetrievedDocOut]
    used_mode: str
    used_temperature: float
    used_max_tokens: int
    fallback: bool


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> Any:
    policy_docs = policy_index.search(req.query, top_k=3)
    playbook_docs = playbook_index.search(req.query, top_k=3)

    policy_top = policy_docs[0].score if policy_docs else 0.0
    playbook_top = playbook_docs[0].score if playbook_docs else 0.0
    best_score = max(policy_top, playbook_top)

    if (not policy_docs and not playbook_docs) or best_score < BM25_MIN_SCORE:
        resp_text = fallback_response()
        log_event(
            logger,
            "generate",
            {
                "query": req.query,
                "mode": req.mode,
                "fallback": True,
                "bm25_top_score": best_score,
                "bm25_policy_top_score": policy_top,
                "bm25_playbook_top_score": playbook_top,
                "retrieved_docs": [],
                "temperature": None,
                "max_tokens": None,
                "prompt_name": "fallback",
            },
        )
        return GenerateResponse(
            response=resp_text,
            retrieved_docs=[],
            used_mode="fallback",
            used_temperature=0.0,
            used_max_tokens=0,
            fallback=True,
        )

    # Minimize LLM tokens: select up to 3 docs total and compact/truncate content.
    selected: list[tuple[str, Any]] = []

    if policy_docs and playbook_docs:
        selected.append(("POLICY", policy_docs[0]))
        selected.append(("DATASET", playbook_docs[0]))

        candidates: list[tuple[float, str, Any]] = []
        if len(policy_docs) > 1:
            denom = policy_top or 1.0
            for d in policy_docs[1:]:
                candidates.append((d.score / denom, "POLICY", d))
        if len(playbook_docs) > 1:
            denom = playbook_top or 1.0
            for d in playbook_docs[1:]:
                candidates.append((d.score / denom, "DATASET", d))

        candidates.sort(key=lambda x: x[0], reverse=True)
        for _, src, d in candidates:
            if len(selected) >= 3:
                break
            selected.append((src, d))

    elif policy_docs:
        selected = [("POLICY", d) for d in policy_docs[:3]]
    else:
        selected = [("DATASET", d) for d in playbook_docs[:3]]

    def _compact(text: str) -> str:
        return " ".join((text or "").split()).strip()

    def _truncate(text: str, max_chars: int) -> str:
        t = _compact(text)
        if len(t) <= max_chars:
            return t
        cut = t[:max_chars]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        return cut + "…"

    formatted_docs: list[RetrievedDocOut] = []
    context_parts: list[str] = []
    for i, (src, d) in enumerate(selected, start=1):
        max_chars = 520 if src == "DATASET" else 650
        content_used = _truncate(d.content, max_chars=max_chars)
        title_used = _compact(d.title)
        context_parts.append(f"[{i}] {src}: {title_used}\n{content_used}")
        formatted_docs.append(
            RetrievedDocOut(
                title=f"[{src}] {title_used}",
                content=content_used,
                score=float(d.score),
            )
        )

    docs_block = "\n\n".join(context_parts)

    prompt, default_temp, default_max_tokens, prompt_name = build_prompt(req.mode, docs_block, req.query)
    temperature = req.temperature if req.temperature is not None else default_temp
    max_tokens = req.max_tokens if req.max_tokens is not None else default_max_tokens

    api_key = os.getenv("SARVAM_API_SUBSCRIPTION_KEY")
    if not api_key:
        raise RuntimeError("Missing environment variable SARVAM_API_SUBSCRIPTION_KEY")

    llm = SarvamLLM(api_subscription_key=api_key)
    resp_text = llm.generate(prompt=prompt, temperature=temperature, max_tokens=max_tokens)
    # Defense-in-depth: ensure frontend never receives internal thought content.
    resp_text = strip_think_tags(resp_text)

    retrieved_out = formatted_docs

    log_event(
        logger,
        "generate",
        {
            "query": req.query,
            "mode": req.mode,
            "fallback": False,
            "bm25_top_score": best_score,
            "bm25_policy_top_score": policy_top,
            "bm25_playbook_top_score": playbook_top,
            "retrieved_docs": [d.model_dump() for d in retrieved_out],
            "prompt_name": prompt_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt": prompt,
        },
    )

    return GenerateResponse(
        response=resp_text,
        retrieved_docs=retrieved_out,
        used_mode=prompt_name,
        used_temperature=temperature,
        used_max_tokens=max_tokens,
        fallback=False,
    )
