import argparse
import os

from dotenv import load_dotenv
from livekit.api import AccessToken, VideoGrants

load_dotenv()

parser = argparse.ArgumentParser(description="Generate a LiveKit access token for testing")
parser.add_argument("--room", default="test-room", help="Room name to join")
parser.add_argument("--identity", default="playground-user", help="Participant identity")
args = parser.parse_args()

token = (
    AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
    .with_identity(args.identity)
    .with_name(args.identity)
    .with_grants(
        VideoGrants(
            room_join=True,
            room=args.room,
            can_publish=True,
            can_subscribe=True,
        )
    )
    .to_jwt()
)

print("LIVEKIT_URL:", os.environ["LIVEKIT_URL"])
print("ROOM:", args.room)
print("IDENTITY:", args.identity)
print("TOKEN:", token)
