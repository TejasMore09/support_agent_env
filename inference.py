"""
Inference Script — Customer Support Agent Environment
======================================================
MANDATORY environment variables:
  API_BASE_URL   The API endpoint for the LLM (e.g. https://router.huggingface.co/v1)
  MODEL_NAME     Model identifier (e.g. Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       Your HuggingFace API key

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

from openai import OpenAI

from my_env.env import CustomerSupportEnv

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN", "")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TASK_NAME = "customer-support"
ENV_NAME  = "customer_support_agent_env"

# ── Logging ────────────────────────────────────────────────────────────────────

def log_start() -> None:
    print(
        f"[START] task={TASK_NAME} env={ENV_NAME} model={MODEL_NAME}",
        flush=True
    )


def log_step(step: int, action: str, reward: float, done: bool, error: str | None = None) -> None:
    # Truncate action to 200 chars to keep log readable
    action_safe = action.replace("\n", " ").replace("\r", " ")[:200]
    err_str = error if error else "null"
    print(
        f"[STEP] step={step} action={action_safe!r} reward={reward:.2f} "
        f"done={str(done).lower()} error={err_str}",
        flush=True
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    r_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={r_str}",
        flush=True
    )

# ── Prompt builders ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a professional customer support agent. "
    "You respond clearly, empathetically, and with concrete next steps. "
    "Always maintain a professional tone."
)

def build_prompt(email_body: str, task: str) -> str:
    if task == "classify":
        return (
            f"Classify the following customer email into EXACTLY ONE category: "
            f"billing, complaint, or query.\n\n"
            f"Email:\n{email_body}\n\n"
            f"Respond with the category name and a one-sentence justification."
        )
    elif task == "respond":
        return (
            f"Write a professional customer support reply to the following email. "
            f"Include: a greeting, an acknowledgement of their issue, an apology if appropriate, "
            f"a concrete next action, and a professional closing.\n\n"
            f"Customer Email:\n{email_body}"
        )
    else:  # resolve
        return (
            f"You are resolving a customer support ticket end-to-end. "
            f"First, state the category (billing/complaint/query). "
            f"Then write a complete, professional reply that fully resolves the issue — "
            f"include apology, concrete resolution steps, timeline, and professional closing.\n\n"
            f"Customer Email:\n{email_body}"
        )

# ── Main agent loop ────────────────────────────────────────────────────────────

async def run_episode(env: CustomerSupportEnv) -> tuple[bool, int, float, list[float]]:
    rewards: list[float] = []
    step_num = 0
    last_error: str | None = None

    try:
        reset_result = await env.reset()
        obs = reset_result["observation"]

        for step_num in range(1, 4):  # 3 tasks max
            email_body = obs["email_body"]
            task       = obs["task"]

            prompt = build_prompt(email_body, task)

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
                error_str = None
            except Exception as api_err:
                reply     = ""
                error_str = str(api_err)
                last_error = error_str

            step_result = await env.step({"reply": reply})
            reward = step_result["reward"]
            done   = step_result["done"]
            rewards.append(reward)

            log_step(step_num, reply, reward, done, error=error_str)

            if done:
                break

            obs = step_result["observation"]

    except Exception as outer_err:
        last_error = str(outer_err)
        # Emit a zero-reward step so [END] is always reachable
        log_step(step_num or 1, "", 0.0, True, error=last_error)
        rewards = rewards or [0.0]

    score   = round(sum(rewards) / len(rewards), 4) if rewards else 0.0
    success = score >= 0.6

    return success, step_num, score, rewards


async def main() -> None:
    env = CustomerSupportEnv()
    log_start()

    success, steps, score, rewards = await run_episode(env)

    await env.close()
    log_end(success, steps, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())