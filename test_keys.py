#
# Quick API key sanity check for the cascade pipeline:
#   STT  -> Deepgram
#   LLM  -> Google Gemini
#   TTS  -> Deepgram
#
# Run with:  uv run test_keys.py
#

import asyncio
import os
import sys

import aiohttp
from dotenv import load_dotenv

load_dotenv(override=True)

GREEN = "\033[92m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"


def ok(label, detail=""):
    print(f"{GREEN}✓{RESET} {label} {DIM}{detail}{RESET}")


def fail(label, detail=""):
    print(f"{RED}✗{RESET} {label} {DIM}{detail}{RESET}")


async def test_deepgram_stt(session):
    """Validate the Deepgram key via a tiny silent-audio transcription request."""
    key = os.getenv("DEEPGRAM_API_KEY")
    if not key:
        fail("Deepgram STT", "DEEPGRAM_API_KEY not set")
        return False

    # 100ms of silence, 16kHz mono 16-bit PCM
    pcm = b"\x00\x00" * 1600
    url = "https://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=16000&channels=1"
    headers = {"Authorization": f"Token {key}", "Content-Type": "audio/raw"}
    try:
        async with session.post(url, headers=headers, data=pcm) as r:
            if r.status == 200:
                ok("Deepgram STT", "key valid")
                return True
            body = await r.text()
            fail("Deepgram STT", f"HTTP {r.status}: {body[:120]}")
            return False
    except Exception as e:
        fail("Deepgram STT", str(e))
        return False


async def test_deepgram_tts(session):
    """Validate the Deepgram key + voice model via a tiny TTS request."""
    key = os.getenv("DEEPGRAM_API_KEY")
    if not key:
        fail("Deepgram TTS", "DEEPGRAM_API_KEY not set")
        return False

    voice = os.getenv("DEEPGRAM_VOICE", "aura-2-thalia-en")
    url = f"https://api.deepgram.com/v1/speak?model={voice}&encoding=linear16&sample_rate=16000"
    headers = {"Authorization": f"Token {key}", "Content-Type": "application/json"}
    try:
        async with session.post(url, headers=headers, json={"text": "Hi."}) as r:
            if r.status == 200:
                audio = await r.read()
                ok("Deepgram TTS", f"voice={voice}, {len(audio)} bytes audio")
                return True
            body = await r.text()
            fail("Deepgram TTS", f"HTTP {r.status}: {body[:120]}")
            return False
    except Exception as e:
        fail("Deepgram TTS", str(e))
        return False


async def test_gemini_llm(session):
    """Validate the Google Gemini key via a minimal generateContent call."""
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        fail("Google Gemini LLM", "GOOGLE_API_KEY not set")
        return False

    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"Content-Type": "application/json", "x-goog-api-key": key}
    payload = {"contents": [{"parts": [{"text": "Say 'ok' and nothing else."}]}]}
    try:
        async with session.post(url, headers=headers, json=payload) as r:
            if r.status == 200:
                data = await r.json()
                text = (
                    data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                    .strip()
                )
                ok("Google Gemini LLM", f"model={model}, reply={text!r}")
                return True
            body = await r.text()
            fail("Google Gemini LLM", f"HTTP {r.status}: {body[:120]}")
            return False
    except Exception as e:
        fail("Google Gemini LLM", str(e))
        return False


async def test_glm_llm(session):
    """Validate the GLM (Zhipu AI) key via its OpenAI-compatible chat endpoint."""
    key = os.getenv("GLM_API_KEY")
    if not key:
        fail("GLM LLM", "GLM_API_KEY not set")
        return False

    model = os.getenv("GLM_MODEL", "glm-4-flash")
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say 'ok' and nothing else."}],
        "max_tokens": 16,
    }
    try:
        async with session.post(url, headers=headers, json=payload) as r:
            if r.status == 200:
                data = await r.json()
                text = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                ok("GLM LLM", f"model={model}, reply={text!r}")
                return True
            body = await r.text()
            fail("GLM LLM", f"HTTP {r.status}: {body[:120]}")
            return False
    except Exception as e:
        fail("GLM LLM", str(e))
        return False


async def test_openai_realtime(session):
    """Validate the OpenAI key + Realtime access by minting an ephemeral token.

    Hitting /v1/realtime/client_secrets confirms the key is valid and has Realtime
    access without opening a WebSocket session.
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        fail("OpenAI Realtime", "OPENAI_API_KEY not set")
        return False

    model = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime")
    url = "https://api.openai.com/v1/realtime/client_secrets"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    payload = {"session": {"type": "realtime", "model": model}}
    try:
        async with session.post(url, headers=headers, json=payload) as r:
            if r.status == 200:
                data = await r.json()
                # Response carries an ephemeral secret (shape varies by API version).
                secret = data.get("value") or data.get("client_secret", {}).get("value", "")
                ok("OpenAI Realtime", f"model={model}, ephemeral token issued ({secret[:12]}…)")
                return True
            body = await r.text()
            fail("OpenAI Realtime", f"HTTP {r.status}: {body[:120]}")
            return False
    except Exception as e:
        fail("OpenAI Realtime", str(e))
        return False


async def main():
    print("Testing API keys for STT, LLM, and TTS...\n")
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            # test_deepgram_stt(session),
            # test_gemini_llm(session),
            # test_glm_llm(session),
            test_openai_realtime(session),
            # test_deepgram_tts(session),
        )
    print()
    if all(results):
        print(f"{GREEN}All keys valid.{RESET}")
        sys.exit(0)
    else:
        print(f"{RED}One or more keys failed — see above.{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
