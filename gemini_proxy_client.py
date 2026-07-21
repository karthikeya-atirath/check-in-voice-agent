"""Gemini client for Vertex AI (asia-south1 / Mumbai), routed through the internal proxy.

No service account and no ADC on this box: the proxy at GEMINI_BASE_URL holds the
Google credential. It only injects that credential when the incoming request has no
Authorization header of its own -- and the genai SDK always signs Vertex requests --
so we strip the header on the way out and let the proxy authenticate for us.
"""
import os

import httpx
from google import genai
from google.oauth2.credentials import Credentials

PROJECT = "qcbot-tcil"
LOCATION = "asia-south1"  # Mumbai


def _strip_auth(request):
    """Drop the SDK's Authorization header so the proxy supplies the real credential."""
    request.headers.pop("Authorization", None)
    request.headers.pop("authorization", None)


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
            "base_url": os.environ["GEMINI_BASE_URL"],
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
