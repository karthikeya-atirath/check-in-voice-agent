"""TTS service for the AgentSession: Cartesia sonic-2, routed through the internal proxy.

The direct Cartesia API key stopped working, so requests go through an internal
proxy that holds the real credential. The plugin always sends whatever api_key
we pass as the `X-API-Key` header, so the proxy is expected to overwrite that
header with the real key before forwarding upstream -- the placeholder here is
never actually checked by Cartesia.
"""
import os

from livekit.plugins import cartesia

MODEL = "sonic-2"


def get_tts() -> cartesia.TTS:
    return cartesia.TTS(
        model=MODEL,
        base_url=os.environ["CARTESIA_BASE_URL"],
        api_key="unused",  # placeholder; proxy supplies the real credential
    )
