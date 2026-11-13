# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamo transport preserves the qualified capture and session contracts."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
from nemo_gym.rollout_correlation import rollout_context
from nemo_gym.token_id_capture.config import TokenIdCaptureConfig
from nemo_gym.token_id_capture.external_capture import (
    DYNAMO_GYM_EXTRAS_ANNOTATION_PREFIX,
    DynamoWorkerCaptureHandler,
    add_dynamo_gym_extras,
    dynamo_capture_refusal_suffix,
    make_external_capture_handler,
    split_dynamo_capture_refusal,
)
from nemo_gym.token_id_capture.lineage import InMemoryLineageStore
from nemo_gym.token_id_capture.sink import CaptureContext, reset_token_sink, set_token_sink
from nemo_gym.token_id_capture.staging.records import CaptureAdmission, CommitCoords
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


def _extras(payload):
    annotation = next(
        item for item in payload["nvext"]["annotations"] if item.startswith(DYNAMO_GYM_EXTRAS_ANNOTATION_PREFIX)
    )
    return json.loads(annotation.removeprefix(DYNAMO_GYM_EXTRAS_ANNOTATION_PREFIX))


def test_dynamo_annotation_merges_without_mutating_shared_config():
    shared = {"annotations": ["other:keep", 'nemo_rl:gym:{"ng_measure":true}']}
    before = copy.deepcopy(shared)
    request = {"nvext": shared}
    add_dynamo_gym_extras(request, {"stop_token_ids": [15]})
    assert shared == before
    assert _extras(request) == {"ng_measure": True, "stop_token_ids": [15]}
    assert request["nvext"]["annotations"][0] == "other:keep"


@pytest.mark.parametrize("nvext", [[], {"annotations": "invalid"}, {"annotations": ["nemo_rl:gym:[]"]}])
def test_invalid_dynamo_annotations_fail_before_dispatch(nvext):
    with pytest.raises(ValueError):
        add_dynamo_gym_extras({"nvext": nvext}, {})


def test_dynamo_preserves_qualified_stop_ids_and_decoding_options():
    request = {
        "messages": [{"role": "assistant", "content": "old", "prompt_token_ids": [1], "generation_token_ids": [2]}],
        "stop": ["</tool_call>"],
        "stop_token_ids": [15],
        "skip_special_tokens": False,
        "include_stop_str_in_output": True,
        "return_tokens_as_token_ids": True,
    }
    VLLMModel._prepare_dynamo_request(SimpleNamespace(_external_capture_handler=None), request)
    assert _extras(request) == {
        "required_prefix_token_ids": [1, 2],
        "stop_token_ids": [15],
        "skip_special_tokens": False,
        "include_stop_str_in_output": True,
    }
    assert request["stop"] == ["</tool_call>"]
    assert request["messages"] == [{"role": "assistant", "content": "old"}]
    assert (
        not {"stop_token_ids", "skip_special_tokens", "include_stop_str_in_output", "return_tokens_as_token_ids"}
        & request.keys()
    )


def test_dynamo_canonicalizes_baseline_reasoning_aliases():
    request = {
        "messages": [{"role": "assistant", "content": "done", "reasoning": "think", "reasoning_content": "think"}]
    }
    VLLMModel._prepare_dynamo_request(SimpleNamespace(_external_capture_handler=None), request)
    assert request["messages"][0]["reasoning_content"] == "think"
    assert "reasoning" not in request["messages"][0]
    request["messages"][0]["reasoning"] = "different"
    with pytest.raises(ValueError, match="conflicting reasoning aliases"):
        VLLMModel._prepare_dynamo_request(SimpleNamespace(_external_capture_handler=None), request)


@pytest.mark.asyncio
@pytest.mark.parametrize("coords_present", [True, False])
async def test_dynamo_capture_commits_only_durable_worker_acknowledgements(coords_present):
    store = InMemoryLineageStore()
    context = CaptureContext(
        rollout_id="r1-a2",
        model_call_id="c1",
        token_sink=None,
        lineage_store=store,
        external_staging=True,
        request_items=[{"role": "user", "content": "go"}],
        capture_admission=CaptureAdmission(rollout_id="r1-a2", model_call_id="c1", mode="text"),
    )
    handler = make_external_capture_handler("dynamo_worker")
    assert isinstance(handler, DynamoWorkerCaptureHandler)
    token = set_token_sink(context)
    try:
        request = handler.prepare_request({"messages": [], "top_logprobs": 3})
        assert _extras(request)["ng_capture"]["rollout_id"] == "r1-a2"
        assert handler.request_headers() == {"x-dynamo-session-id": "r1-a2"}
        assert request["logprobs"] is True and request["top_logprobs"] == 3
        coords = CommitCoords(
            rollout_id="r1-a2",
            model_call_id="c1",
            prev_len=0,
            weight_version=7,
            delta_len=3,
            cum_len=3,
            digest="0" * 64,
            extras_digest="1" * 64,
            staging_key="r1-a2/c1",
            chain_hash="2" * 64,
            cumulative_hash="3" * 64,
        ).model_dump(mode="json")
        payload = {
            "id": "response1",
            "choices": [{"message": {"role": "assistant", "content": "done"}}],
            "nvext": {"nemo_rl": {"ng_commit_coords": coords} if coords_present else {}},
        }
        handler.prepare_response(payload)
        assert "nvext" not in payload and "ng_commit_coords" not in payload
        await handler.finalize_response(payload)
    finally:
        reset_token_sink(token)
    manifest = await store.manifest("r1-a2")
    assert context.committed is coords_present
    if coords_present:
        assert manifest["records"][0]["staging_key"] == "r1-a2/c1"
        assert manifest["records"][0]["weight_version"] == 7
        assert manifest["failures"] == []
    else:
        assert manifest["records"] == []
        assert manifest["failures"][0]["reason"] == "worker_response_missing_commit_coordinates"


@pytest.mark.asyncio
async def test_session_header_is_attempt_scoped_and_capture_override_wins(monkeypatch):
    client = NeMoGymAsyncOpenAI(
        base_url="http://core/v1",
        api_key="unused",
        default_headers={"X-Dynamo-Session-ID": "stale"},
        session_key_headers=["x-dynamo-session-id"],
    )
    send = AsyncMock(return_value=object())
    monkeypatch.setattr(NeMoGymAsyncOpenAI, "_request_with_retry", send)
    with rollout_context("attempt-a1"):
        await client._request(url="http://core/v1/chat/completions", method="POST")
    headers = send.call_args.kwargs["headers"]
    assert {name.lower(): value for name, value in headers.items()}["x-dynamo-session-id"] == "attempt-a1"
    with rollout_context("attempt-a2"):
        await client._request(
            url="http://core/v1/chat/completions", method="POST", headers={"X-DYNAMO-SESSION-ID": "capture-owner"}
        )
    headers = send.call_args.kwargs["headers"]
    assert sum(name.lower() == "x-dynamo-session-id" for name in headers) == 1
    assert {name.lower(): value for name, value in headers.items()}["x-dynamo-session-id"] == "capture-owner"


@pytest.mark.asyncio
async def test_capture_headers_are_http_metadata_not_openai_body(monkeypatch):
    client = NeMoGymAsyncOpenAI(base_url="http://core/v1", api_key="unused")
    response = SimpleNamespace(ok=True)
    request = AsyncMock(return_value=response)
    monkeypatch.setattr(NeMoGymAsyncOpenAI, "_request", request)
    monkeypatch.setattr("nemo_gym.openai_utils.get_response_json", AsyncMock(return_value={"id": "answer"}))
    monkeypatch.setattr("nemo_gym.openai_utils.raise_for_status", AsyncMock())
    await client.create_chat_completion(_headers={"x-dynamo-session-id": "r1"}, messages=[])
    assert request.call_args.kwargs["headers"] == {"x-dynamo-session-id": "r1"}
    assert request.call_args.kwargs["json"] == {"messages": []}


def test_dynamo_refusal_suffix_preserves_error_and_call_identity():
    message = "context length exceeded" + dynamo_capture_refusal_suffix("c1")
    assert split_dynamo_capture_refusal(message) == ("context length exceeded", "c1")
    assert split_dynamo_capture_refusal("ordinary failure") == ("ordinary failure", None)


def test_dynamo_external_staging_requires_capture_enabled():
    with pytest.raises(ValueError, match="requires external_staging"):
        TokenIdCaptureConfig(token_id_capture={"external_staging_backend": "dynamo_worker"})


@pytest.mark.parametrize(
    "field", ["use_completions_api", "is_responses_native", "request_prompt_and_generation_token_ids"]
)
def test_dynamo_rejects_incompatible_openai_routes(field):
    with pytest.raises(ValueError, match="Dynamo supports chat"):
        VLLMModelConfig(
            host="127.0.0.1",
            port=5000,
            entrypoint="app.py",
            name="policy",
            base_url="http://core/v1",
            api_key="unused",
            model="nano",
            return_token_id_information=False,
            uses_reasoning_parser=False,
            dynamo_frontend=True,
            **{field: True},
        )
