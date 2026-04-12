"""
Inference Script — Customer Support Agent Environment
======================================================
MANDATORY environment variables (injected by validator):
  API_BASE_URL   The LiteLLM proxy endpoint
  API_KEY        The proxy API key
  MODEL_NAME     Model identifier (optional)

This script:
1. Uses the OpenAI client with API_BASE_URL and API_KEY exactly as injected
2. Talks to the environment via HTTP (localhost:7860) for reset/step
3. Emits [START]/[STEP]/[END] to stdout
"""

import asyncio
import os
import sys
import json
import urllib.request
import urllib.error

# ── sys.path fix ───────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ── Task identity ──────────────────────────────────────────────────────────────
TASK_NAME = "customer-support"
ENV_NAME  = "customer_support_agent_env"

# ── Logging ────────────────────────────────────────────────────────────────────

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

# ── Read env vars exactly as validator injects them ────────────────────────────
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY      = os.environ["API_KEY"]
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# ── Emit [START] right away ────────────────────────────────────────────────────
log_start(MODEL_NAME)

# ── HTTP helpers to talk to the env server (localhost:7860) ───────────────────
ENV_SERVER = os.environ.get("ENV_SERVER_URL", "http://localhost:7860")

def http_post(path: str, body: dict = None) -> dict:
    url  = f"{ENV_SERVER}{path}"
    data = json.dumps(body or {}).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

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
            "Include: a greeting, acknowledgement of the issue, an apology, "
            "a concrete next action, and a professional closing.\n\n"
            f"Customer Email:\n{email_body}"
        )
    else:
        return (
            "Resolve this customer support ticket end-to-end. "
            "State the category (billing/complaint/query), then write a complete "
            "professional reply with apology, resolution steps, timeline, and closing.\n\n"
            f"Customer Email:\n{email_body}"
        )

# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    from openai import OpenAI

    # Exactly as validator requires
    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"],
    )

    rewards  = []
    step_num = 0

    try:
        # Call env server to reset
        reset_result = http_post("/reset")
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

            # Call env server to step
            step_result = http_post("/step", {"reply": reply})
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
    log_end(success, max(step_num, 1), score, rewards)


if __name__ == "__main__":
    try:
        main()
    except Exception as fatal:
        log_step(1, "", 0.0, True, error=str(fatal))
        log_end(False, 1, 0.0, [0.0])
        sys.exit(0)