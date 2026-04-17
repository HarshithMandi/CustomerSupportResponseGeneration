from __future__ import annotations


def build_prompt(mode: str, docs: str, query: str) -> tuple[str, float, int, str]:
    """Return (prompt, temperature, max_tokens, prompt_name)."""

    mode_norm = (mode or "").strip().lower()

    # Keep prompts short to minimize token usage; detailed constraints live in the system message.
    if mode_norm in {"strict", "strict policy", "strict_policy"}:
        prompt = (
            "Mode: STRICT.\n"
            "Write like a customer support executive (professional, neutral, policy-aligned).\n"
            "Use ONLY the context below. If info is missing, ask for it (order ID, photos) instead of guessing.\n"
            "Format: brief apology + policy-based action + 2–3 bullet next steps + closing (Regards, Customer Support).\n"
            "Keep it short (<= 110 words).\n\n"
            f"Context:\n{docs}\n\n"
            f"Customer message:\n{query}\n\n"
            "Reply:"
        )
        return prompt, 0.2, 150, "strict"

    if mode_norm in {"friendly", "friendly tone", "friendly_tone"}:
        prompt = (
            "Mode: FRIENDLY.\n"
            "Write like a customer support executive (warm, empathetic, policy-aligned).\n"
            "Use ONLY the context below. If info is missing, ask for it (order ID, photos) instead of guessing.\n"
            "Format: brief apology + policy-based action + 2–3 bullet next steps + closing (Regards, Customer Support).\n"
            "Keep it short (<= 130 words).\n\n"
            f"Context:\n{docs}\n\n"
            f"Customer message:\n{query}\n\n"
            "Reply:"
        )
        return prompt, 0.7, 200, "friendly"

    # Default to strict if unknown
    prompt = (
        "Mode: STRICT.\n"
        "Write like a customer support executive (professional, neutral, policy-aligned).\n"
        "Use ONLY the context below. If info is missing, ask for it (order ID, photos) instead of guessing.\n"
        "Format: brief apology + policy-based action + 2–3 bullet next steps + closing (Regards, Customer Support).\n"
        "Keep it short (<= 110 words).\n\n"
        f"Context:\n{docs}\n\n"
        f"Customer message:\n{query}\n\n"
        "Reply:"
    )
    return prompt, 0.2, 150, "strict"


def fallback_response() -> str:
    return "Please escalate this issue to a human support agent."
