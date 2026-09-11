"""Shared Hugging Face dataset configuration for the recorder and web UI."""

import json


DEFAULT_HF_NAMESPACE = "MicroAGI-Labs"
DEFAULT_TASK_PROMPT = (
    "Put all the objects on the table into the gray tote, then carry the tote to the blue "
    "roller conveyor and place it on the conveyor."
)
DATASET_CONFIG_PREFIX = "dataset_config:"


def encode_dataset_config(repo_id: str, prompt: str, private: bool) -> str:
    """Encode an idempotent recorder configuration message for the ZMQ command bus."""
    return DATASET_CONFIG_PREFIX + json.dumps(
        {"repo_id": repo_id, "prompt": prompt, "private": private},
        separators=(",", ":"),
    )


def decode_dataset_config(message: str) -> dict[str, object]:
    """Decode and minimally validate a recorder configuration message."""
    if not message.startswith(DATASET_CONFIG_PREFIX):
        raise ValueError("not a dataset configuration message")
    payload = json.loads(message.removeprefix(DATASET_CONFIG_PREFIX))
    repo_id = payload.get("repo_id")
    prompt = payload.get("prompt")
    private = payload.get("private", True)
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("repo_id must be a non-empty string")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(private, bool):
        raise ValueError("private must be a boolean")
    return {"repo_id": repo_id.strip(), "prompt": prompt.strip(), "private": private}
