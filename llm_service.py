"""LLM service for the AgentSession: Gemini on Vertex AI via the internal proxy."""
from google.oauth2.credentials import Credentials
from livekit.plugins import google

from gemini_proxy_client import LOCATION, PROJECT, build_client

MODEL = "gemini-3.5-flash"


def get_llm() -> google.LLM:
    llm = google.LLM(
        model=MODEL,
        vertexai=True,
        project=PROJECT,
        location=LOCATION,
        credentials=Credentials(token="unused"),
    )
    llm._client = build_client()
    return llm
