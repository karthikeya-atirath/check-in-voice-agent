"""STT service for the AgentSession: Deepgram nova-2, routed through the internal proxy.

The direct Deepgram API key stopped working, so requests go through an internal
proxy that holds the real credential. The plugin always sends whatever api_key
we pass as the `Authorization: Token ...` header, so the proxy is expected to
overwrite that header with the real key before forwarding upstream -- the
placeholder here is never actually checked by Deepgram.
"""
import os

from livekit.plugins import deepgram

MODEL = "nova-2"


def get_stt() -> deepgram.STT:
    return deepgram.STT(
        model=MODEL,
        base_url=os.environ["DEEPGRAM_BASE_URL"],
        api_key="unused",  # placeholder; proxy supplies the real credential
    )
