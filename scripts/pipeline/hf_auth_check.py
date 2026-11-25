"""Check the runtime HF_TOKEN has write access. Never logs or stores the token (no `hf auth login`)."""
import os
import sys

from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN")
if not token:
    sys.exit("HF_TOKEN is not set in the environment")
auth = HfApi(token=token).whoami().get("auth", {}).get("accessToken", {})
role = auth.get("role")
if role == "read":
    sys.exit("HF_TOKEN has read-only access; write access is needed for private model repos")
print(f"HF auth OK (token role: {role})")
