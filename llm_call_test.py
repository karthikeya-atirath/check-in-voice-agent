"""Test Gemini on Vertex AI (asia-south1 / Mumbai) through the internal proxy.

No service account and no ADC on this box: the proxy at GEMINI_BASE_URL holds the
Google credential. It only injects that credential when the incoming request has no
Authorization header of its own -- and the genai SDK always signs Vertex requests --
so we strip the header on the way out and let the proxy authenticate for us.
"""
import asyncio
import math
import time

import httpx
from google import genai
from google.genai import types
from google.oauth2.credentials import Credentials

BASE_URL = "http://10.160.0.5:8007"
PROJECT = "qcbot-tcil"
LOCATION = "asia-south1"  # Mumbai
MODEL = "gemini-3.5-flash"
EMBED_MODEL = "text-embedding-005"


def _strip_auth(request):
    """Drop the SDK's Authorization header so the proxy supplies the real credential."""
    request.headers.pop("Authorization", None)
    request.headers.pop("authorization", None)


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


async def _strip_auth_async(request):
    # httpx requires async event hooks on AsyncClient; a sync callable is silently
    # awaited-as-None and the header survives -> 401.
    _strip_auth(request)


def build_client() -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=PROJECT,
        location=LOCATION,
        credentials=Credentials(token="unused"),  # placeholder; proxy replaces it
        http_options={
            "base_url": BASE_URL,
            # sync  (client.models.*)
            "client_args": {"event_hooks": {"request": [_strip_auth]}},
            # async (client.aio.models.*). The explicit transport is load-bearing:
            # without it the SDK sends async requests over aiohttp, which ignores
            # these httpx event hooks, and the Authorization header survives -> 401.
            "async_client_args": {
                "transport": httpx.AsyncHTTPTransport(),
                "event_hooks": {"request": [_strip_auth_async]},
            },
        },
    )


async def main() -> None:
    print(f"vertex={PROJECT} @ {LOCATION} (Mumbai) via {BASE_URL}")
    print(f"model={MODEL}\n")
    client = build_client()

    # 1. sync — used by generate_call_summary / the judge scorer
    start = time.perf_counter()
    resp = client.models.generate_content(
        model=MODEL,
        contents="Reply with the single word OK.",
        config=types.GenerateContentConfig(max_output_tokens=20),
    )
    print(f"sync   {time.perf_counter() - start:5.2f}s -> {resp.text!r}")

    # 2. async — used by gemini_call (suggestions, params, package search)
    start = time.perf_counter()
    resp = await client.aio.models.generate_content(
        model=MODEL,
        contents="Reply with the single word OK.",
        config=types.GenerateContentConfig(max_output_tokens=20),
    )
    print(f"async  {time.perf_counter() - start:5.2f}s -> {resp.text!r}")

    # 3. streaming, with thinking disabled (the latency-sensitive path)
    start = time.perf_counter()
    first = None
    chunks = []
    for chunk in client.models.generate_content_stream(
        model=MODEL,
        contents="so if i travel faster than light and look ack at the earth will i see past evnt as the light is not fast enough to reach my location.",
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
        ),
    ):
        if chunk.text:
            if first is None:
                first = time.perf_counter() - start
            chunks.append(chunk.text)
    total = time.perf_counter() - start
    print(f"stream {total:5.2f}s (first chunk {first:.2f}s)\n")
    print("".join(chunks))

    # 4. embeddings — used only by the semantic half of end-of-call quality scoring
    print(f"\n--- embeddings ({EMBED_MODEL}) ---")
    start = time.perf_counter()
    suggestion = "Ask the customer what dates they are planning to travel."
    agent_said = "Sure, what dates were you looking at for the trip?"
    unrelated = "The visa fee for Dubai is 6500 rupees."
    try:
        vecs = [
            client.models.embed_content(model=EMBED_MODEL, contents=t).embeddings[0].values
            for t in (suggestion, agent_said, unrelated)
        ]
        dt = time.perf_counter() - start
        print(f"embed  {dt:5.2f}s -> dims={len(vecs[0])} (3 texts)")
        print(f"  cosine(suggestion, agent_said) = {_cosine(vecs[0], vecs[1]):.3f}   (expect high)")
        print(f"  cosine(suggestion, unrelated)  = {_cosine(vecs[0], vecs[2]):.3f}   (expect low)")
        print(f"  first 5 dims: {[round(v, 5) for v in vecs[0][:5]]}")
    except Exception as exc:
        msg = str(exc).replace("\n", " ")
        blocked = "allowedModels" in msg
        print(f"embed  FAILED{' (org policy: constraints/vertexai.allowedModels)' if blocked else ''}")
        print(f"  {msg[:300]}")


if __name__ == "__main__":
    asyncio.run(main())
