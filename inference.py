"""
Inference Script — Customer Support Agent Environment
======================================================
MANDATORY environment variables:
  API_BASE_URL   The API endpoint for the LLM (e.g. https://router.huggingface.co/v1)
  MODEL_NAME     Model identifier (e.g. Qwen/Qwen2.5-72B-Instruct)
  API_KEY       Your HuggingFace API key

STDOUT FORMAT (strictly followed):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Rules:
  - One [START] per episode.
  - One [STEP] per env.step() call, immediately after it returns.
  - One [END] after env.close(), always emitted (even on exception).
  - reward and score formatted to 2 decimal places.
  - done and success lowercase booleans.
  - error is the raw exception string or null.
  - All fields on a single line, no newlines within a line.
  - Each task returns score in [0, 1].
"""

import asyncio
import os
import sys

# ── Fix sys.path FIRST so my_env can be found regardless of working directory ──
# The validator may run this script from /tmp/workspace/ or any other cwd.
# We ensure the directory containing inference.py is always on the path.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ── Constants ──────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
# Validator injects API_KEY; fall back to HF_TOKEN for local runs
API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN", "dummy-token")

TASK_NAME = "customer-support"
ENV_NAME  = "customer_support_agent_env"

# ── Logging — defined before ANYTHING else so they can never fail ──────────────

def log_start() -> None:
    print(f"[START] task={TASK_NAME} env={ENV_NAME} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None) -> None:
    action_safe = str(action).replace("\n", " ").replace("\r", " ")[:200]
    err_str = str(error) if error else "null"
    print(
        f"[STEP] step={step} action={action_safe!r} reward={reward:.2f} "
        f"done={str(done).lower()} error={err_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    r_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={r_str}",
        flush=True,
    )

# ── Emit [START] immediately — before any import that could fail ───────────────
# This guarantees the validator sees at least one structured line.
log_start()

# ── Now import potentially-failing dependencies ────────────────────────────────
try:
    from openai import OpenAI
    _openai_ok = True
except Exception as _e:
    _openai_ok = False
    print(f"# openai import failed: {_e}", file=sys.stderr, flush=True)

try:
    from my_env.env import CustomerSupportEnv
    _env_ok = True
except Exception as _e:
    _env_ok = False
    print(f"# my_env import failed: {_e}", file=sys.stderr, flush=True)

# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a professional customer support agent. "
    "You respond clearly, empathetically, and with concrete next steps. "
    "Always maintain a professional tone."
)


def build_prompt(email_body: str, task: str) -> str:
    if task == "classify":
        return (
            "Classify the following customer email into EXACTLY ONE category: "
            "billing, complaint, or query.\n\n"
            f"Email:\n{email_body}\n\n"
            "Respond with the category name and a one-sentence justification."
        )
    elif task == "respond":
        return (
            "Write a professional customer support reply to the following email. "
            "Include: a greeting, acknowledgement of their issue, an apology if appropriate, "
            "a concrete next action, and a professional closing.\n\n"
            f"Customer Email:\n{email_body}"
        )
    else:  # resolve
        return (
            "You are resolving a customer support ticket end-to-end. "
            "First, state the category (billing/complaint/query). "
            "Then write a complete professional reply that fully resolves the issue — "
            "include apology, concrete resolution steps, timeline, and professional closing.\n\n"
            f"Customer Email:\n{email_body}"
        )

# ── Agent loop ─────────────────────────────────────────────────────────────────

async def run_episode(env, client) -> tuple:
    rewards  = []
    step_num = 0

    try:
        reset_result = await env.reset()
        obs = reset_result["observation"]

        for step_num in range(1, 4):
            email_body = obs["email_body"]
            task       = obs["task"]
            prompt     = build_prompt(email_body, task)
            error_str  = None
            reply      = ""

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=512,
                )
                reply = response.choices[0].message.content.strip()
            except Exception as api_err:
                error_str = str(api_err)

            step_result = await env.step({"reply": reply})
            reward = float(step_result["reward"])
            done   = bool(step_result["done"])
            rewards.append(reward)

            log_step(step_num, reply, reward, done, error=error_str)

            if done:
                break

            obs = step_result["observation"]

    except Exception as outer_err:
        if not rewards:
            log_step(max(step_num, 1), "", 0.0, True, error=str(outer_err))
            rewards = [0.0]

    score   = round(sum(rewards) / len(rewards), 4) if rewards else 0.0
    success = score >= 0.6
    return success, max(step_num, 1), score, rewards


async def main() -> None:
    # If imports failed, emit dummy steps so output parsing still sees structure
    if not _env_ok or not _openai_ok:
        log_step(1, "", 0.0, True, error="import-failed")
        log_step(2, "", 0.0, True, error="import-failed")
        log_step(3, "", 0.0, True, error="import-failed")
        log_end(False, 3, 0.0, [0.0, 0.0, 0.0])
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env    = CustomerSupportEnv()

    success, steps, score, rewards = await run_episode(env, client)

    try:
        await env.close()
    except Exception:
        pass

    log_end(success, steps, score, rewards)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as fatal:
        log_step(1, "", 0.0, True, error=str(fatal))
        log_end(False, 1, 0.0, [0.0])
        sys.exit(0)