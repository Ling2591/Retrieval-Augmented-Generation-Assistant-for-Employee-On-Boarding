# querying/generate.py
# PHASE 2, step 6. The only file that talks to Groq.
#
#   generate(messages)   send messages, get the answer text plus usage numbers
#   get_api_key()        finds your key (see below)
#
# Where the key comes from, in this order.
#   1. the environment variable GROQ_API_KEY
#   2. the line  GROQ_API_KEY = "gsk_..."  in .streamlit/secrets.toml
# Never type the key into a code file.

import os
import re
import time

import groq

import config

_client = None


def get_api_key():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key

    if config.SECRETS_PATH.exists():
        text = config.SECRETS_PATH.read_text(encoding="utf-8-sig")
        match = re.search(r"^\s*GROQ_API_KEY\s*=\s*[\"']([^\"']+)[\"']", text, re.MULTILINE)
        if match:
            return match.group(1).strip()

    raise RuntimeError(
        "No Groq API key found. Create the file "
        f"{config.SECRETS_PATH} containing one line shaped like  GROQ_API_KEY = \"your key\"  "
        "or set the environment variable GROQ_API_KEY."
    )


def _get_client():
    """Create the Groq client once. Its own automatic retries are switched off
    because generate() handles retries itself and explains what is happening."""
    global _client
    if _client is None:
        _client = groq.Groq(api_key=get_api_key(), max_retries=0, timeout=config.LLM_TIMEOUT_SECONDS)
    return _client


def _seconds_to_wait(error, fallback):
    """Groq says how long to wait in the retry-after header, when it sends one."""
    try:
        return float(error.response.headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return fallback


def generate(messages, max_tokens=None):
    """Send the messages to Groq and return a dictionary.
       text               the answer
       model              the model that answered
       prompt_tokens, completion_tokens, total_tokens
       seconds            how long the successful call took
    Busy servers and rate limits are retried with a growing pause."""
    client = _get_client()

    options = {}
    if config.LLM_REASONING_EFFORT:                 # only models that think, such as gpt-oss
        options["reasoning_effort"] = config.LLM_REASONING_EFFORT
        options["include_reasoning"] = False         # keep the hidden thinking out of the reply

    pause = 5.0
    for attempt in range(1, config.LLM_MAX_RETRIES + 1):
        started = time.time()
        try:
            response = client.chat.completions.create(
                model=config.GROQ_MODEL_NAME,
                messages=messages,
                temperature=config.LLM_TEMPERATURE,
                max_completion_tokens=max_tokens or config.LLM_MAX_OUTPUT_TOKENS,
                **options,
            )
        except groq.APIStatusError as error:
            retryable = error.status_code == 429 or error.status_code >= 500
            if not retryable:
                raise RuntimeError(
                    f"Groq rejected the request (status {error.status_code}). {error}"
                ) from error
            wait = _seconds_to_wait(error, pause)
            if wait > config.MAX_WAIT_SECONDS:
                raise RuntimeError(
                    f"Groq asked for a wait of {wait:.0f} seconds, which usually means a daily "
                    "limit was reached. Try again later, or switch to the other gpt-oss model "
                    "in config.py."
                ) from error
            if attempt == config.LLM_MAX_RETRIES:
                raise RuntimeError("Groq was still busy after several tries.") from error
            print(f"  Groq is busy (status {error.status_code}). Waiting {wait:.0f} seconds, try {attempt + 1}.")
            time.sleep(wait)
            pause *= 2
            continue
        except groq.APIConnectionError as error:
            if attempt == config.LLM_MAX_RETRIES:
                raise RuntimeError(f"Could not reach Groq. Check your internet connection. {error}") from error
            print(f"  Connection problem. Waiting {pause:.0f} seconds, try {attempt + 1}.")
            time.sleep(pause)
            pause *= 2
            continue

        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        if not text:
            raise RuntimeError(
                f"The model returned an empty answer (finish reason {choice.finish_reason}). "
                "If the reason is length, raise LLM_MAX_OUTPUT_TOKENS in config.py."
            )
        if choice.finish_reason == "length":
            print("  WARNING this answer was cut off at the token limit. Raise LLM_MAX_OUTPUT_TOKENS in config.py.")
        usage = response.usage
        return {
            "text": text,
            "model": response.model,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
            "seconds": time.time() - started,
        }

    raise RuntimeError("Groq did not answer.")      # not reachable, kept as a safety net


# ---------------------------------------------------------------------------
# Self test. Run with  python -m querying.generate
# 1. checks your key works by listing the models your account can use
# 2. checks the model named in config.py is on that list
# 3. sends one tiny question
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    client = _get_client()

    available = sorted(model.id for model in client.models.list().data)
    print("Models available to your account")
    for name in available:
        marker = "   <-- chosen in config.py" if name == config.GROQ_MODEL_NAME else ""
        print(f"  {name}{marker}")
    if config.GROQ_MODEL_NAME not in available:
        print(f"\nWARNING {config.GROQ_MODEL_NAME} is not on the list. Pick a name from the list and edit GROQ_MODEL_NAME in config.py.")
        raise SystemExit(1)

    result = generate([{"role": "user", "content": "Reply with the single word OK."}], max_tokens=200)
    print(f"\nTest call worked in {result['seconds']:.1f} seconds")
    print("Reply ", result["text"])
    print("Tokens", result["total_tokens"], "(prompt", result["prompt_tokens"], "+ answer", result["completion_tokens"], ")")