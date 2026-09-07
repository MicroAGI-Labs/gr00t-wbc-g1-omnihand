"""Versioned topic-prefixed MessagePack contracts for external hands."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import msgpack

HAND_INTENT_TOPIC = b"hand_intent"
HAND_CONFIG_TOPIC = b"hand_config"
HAND_STATE_TOPIC = b"hand_state"
HAND_SIM_FEEDBACK_TOPIC = b"hand_sim_feedback"
HAND_INTENT_SCHEMA = "sonic.hand_intent.v1"
HAND_CONFIG_SCHEMA = "sonic.hand_config.v1"
HAND_STATE_SCHEMA = "sonic.hand_state.v1"
HAND_SIM_FEEDBACK_SCHEMA = "sonic.hand_sim_feedback.v1"
HAND_CONTROL_TOPIC = b"hand_control"
HAND_CONTROL_SCHEMA = "sonic.hand_control.v1"


class HandProtocolError(ValueError):
    pass


def encode(topic: bytes | str, payload: Mapping[str, Any]) -> bytes:
    topic_bytes = topic.encode() if isinstance(topic, str) else topic
    if not topic_bytes or b" " in topic_bytes:
        raise HandProtocolError("topic must be a non-empty token")
    return topic_bytes + b" " + msgpack.packb(dict(payload), use_bin_type=True)


def decode(raw: bytes, topic: bytes | str, schema: str) -> dict[str, Any]:
    topic_bytes = topic.encode() if isinstance(topic, str) else topic
    prefix = topic_bytes + b" "
    if not raw.startswith(prefix):
        raise HandProtocolError(f"message is not on topic {topic_bytes.decode(errors='replace')}")
    try:
        payload = msgpack.unpackb(raw[len(prefix) :], raw=False, strict_map_key=False)
    except Exception as exc:
        raise HandProtocolError(f"invalid MessagePack: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        actual = payload.get("schema") if isinstance(payload, dict) else None
        raise HandProtocolError(f"expected schema {schema}, got {actual!r}")
    return payload


def decode_control(raw: bytes) -> dict[str, Any]:
    payload = decode(raw, HAND_CONTROL_TOPIC, HAND_CONTROL_SCHEMA)
    if payload.get("action") != "reconnect":
        raise HandProtocolError("unsupported hand control action")
    return payload


def decode_intent(raw: bytes) -> dict[str, Any]:
    payload = decode(raw, HAND_INTENT_TOPIC, HAND_INTENT_SCHEMA)
    for field in ("sequence", "monotonic_ns"):
        if isinstance(payload.get(field), bool) or not isinstance(payload.get(field), int):
            raise HandProtocolError(f"{field} must be an integer")
    if payload.get("source") != "pico":
        raise HandProtocolError("hand intent source must be 'pico'")
    for side in ("left", "right"):
        value = payload.get(side)
        if not isinstance(value, dict):
            raise HandProtocolError(f"{side} intent is missing")
        if not isinstance(value.get("valid"), bool) or not isinstance(value.get("closed"), bool):
            raise HandProtocolError(f"{side} valid/closed must be booleans")
        trigger = value.get("trigger")
        if isinstance(trigger, bool) or not isinstance(trigger, (int, float)) or not 0.0 <= float(trigger) <= 1.0:
            raise HandProtocolError(f"{side} trigger must be in [0, 1]")
    return payload


def decode_config(raw: bytes) -> dict[str, Any]:
    return decode(raw, HAND_CONFIG_TOPIC, HAND_CONFIG_SCHEMA)


def decode_state(raw: bytes) -> dict[str, Any]:
    return decode(raw, HAND_STATE_TOPIC, HAND_STATE_SCHEMA)


def decode_sim_feedback(raw: bytes) -> dict[str, Any]:
    return decode(raw, HAND_SIM_FEEDBACK_TOPIC, HAND_SIM_FEEDBACK_SCHEMA)
