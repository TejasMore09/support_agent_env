"""
Inference Script — Customer Support Agent Environment
======================================================
MANDATORY environment variables (injected by validator):
  API_BASE_URL   The API endpoint for the LLM proxy
  API_KEY        The API key for the LLM proxy
  MODEL_NAME     Model identifier

STDOUT FORMAT:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import sys

# ── Fix sys.path so my_env is importable regardless of working directory ───────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ── Task identity (no env vars needed — safe at module level) ──────────────────
TASK_NAME = "customer-support"
ENV_NAME  = "customer_support_agent_env"

# ── Logging functions — defined first, called first ────────────────────────────

def log_start(model: str) -> None:
    print(f"[START] task={TASK_NAME} env={ENV_NAME} model={model}", flush=True)


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

# ── Read env vars (validator always injects these) ─────────────────────────────
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY      = os.environ["API_KEY"]
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# ── Emit [START] immediately after env vars are read ──────────────────────────
log_start(MODEL_NAME)

# ── Import dependencies ────────────────────────────────────────────────────────
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
    if not _env_ok or not _openai_ok:
        log_step(1, "", 0.0, True, error="import-failed")
        log_step(2, "", 0.0, True, error="import-failed")
        log_step(3, "", 0.0, True, error="import-failed")
        log_end(False, 3, 0.0, [0.0, 0.0, 0.0])
        return

    # Exactly as validator requires: base_url and api_key from env vars
    client = OpenAI(base_url=os.environ["API_BASE_URL"], api_key=os.environ["API_KEY"])
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