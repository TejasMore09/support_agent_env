import os
import sys
import asyncio

# sys.path fix
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

# Read EXACTLY as validator injects - no fallbacks
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY      = os.environ["API_KEY"]
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

TASK_NAME = "customer-support"
ENV_NAME  = "customer_support_agent_env"

def log_start(model):
    print(f"[START] task={TASK_NAME} env={ENV_NAME} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    a = str(action).replace("\n"," ").replace("\r","")[:200]
    e = str(error) if error else "null"
    print(f"[STEP] step={step} action={a!r} reward={reward:.2f} done={str(done).lower()} error={e}", flush=True)

def log_end(success, steps, score, rewards):
    r = ",".join(f"{x:.2f}" for x in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={r}", flush=True)

log_start(MODEL_NAME)

SYSTEM_PROMPT = "You are a professional customer support agent. Respond clearly, empathetically, with concrete next steps."

def build_prompt(email_body, task):
    if task == "classify":
        return f"Classify this email into exactly one of: billing, complaint, query.\n\nEmail: {email_body}\n\nRespond with just the category name."
    elif task == "respond":
        return f"Write a professional customer support reply.\n\nCustomer Email: {email_body}"
    else:
        return f"Fully resolve this customer support ticket. State category, then write complete professional reply with apology, resolution steps, and closing.\n\nEmail: {email_body}"

async def main():
    from openai import OpenAI
    from my_env.env import CustomerSupportEnv

    # Exactly as validator specifies
    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"],
    )

    env = CustomerSupportEnv()
    rewards = []
    step_num = 0

    try:
        result = await env.reset()
        obs = result["observation"]

        for step_num in range(1, 4):
            prompt = build_prompt(obs["email_body"], obs["task"])
            reply = ""
            error_str = None

            try:
                resp = client.chat.completions.create(
                    model=os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"),
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=512,
                )
                reply = resp.choices[0].message.content.strip()
            except Exception as e:
                error_str = str(e)

            result = await env.step({"reply": reply})
            reward = float(result["reward"])
            done = bool(result["done"])
            rewards.append(reward)

            log_step(step_num, reply, reward, done, error=error_str)

            if done:
                break
            obs = result["observation"]

    except Exception as e:
        if not rewards:
            log_step(max(step_num,1), "", 0.0, True, error=str(e))
            rewards = [0.0]

    try:
        await env.close()
    except:
        pass

    score = round(sum(rewards)/len(rewards), 4) if rewards else 0.0
    log_end(score >= 0.6, max(step_num,1), score, rewards)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        log_step(1, "", 0.0, True, error=str(e))
        log_end(False, 1, 0.0, [0.0])
        sys.exit(0)