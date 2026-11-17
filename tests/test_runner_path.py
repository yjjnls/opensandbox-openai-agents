"""Public Runner path test: client-owned session lifecycle through Runner.run.

Uses `ScriptedModel` so no model API key is needed, and patches the OpenSandbox SDK
entry points so no sandbox service is contacted.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from agents import RunConfig, Runner
from agents.sandbox import SandboxAgent, SandboxRunConfig
from agents.testing import ScriptedModel

import fakes
from opensandbox_agents import OpenSandboxClient, OpenSandboxClientOptions


def _final_message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="1",
        type="message",
        role="assistant",
        content=[
            ResponseOutputText(text=text, type="output_text", annotations=[], logprobs=[])
        ],
        status="completed",
    )


async def test_runner_owns_client_session_lifecycle() -> None:
    model = ScriptedModel(steps=[[_final_message("done")]])
    client = OpenSandboxClient(api_key="key:secret", domain="sandbox.example.com")
    fake = fakes.make_fake_sandbox("sbx-runner")

    with patch("opensandbox.Sandbox.create", new=AsyncMock(return_value=fake)):
        agent = SandboxAgent(
            name="sandboxed",
            model=model,
        )
        result = await Runner.run(
            agent,
            "say done",
            run_config=RunConfig(
                sandbox=SandboxRunConfig(
                    client=client,
                    options=OpenSandboxClientOptions(image="img:1"),
                ),
            ),
        )

    assert result.final_output == "done"
    # Session startup prepares the workspace root through the provider.
    first_command = fake.commands.calls[0][0]
    assert first_command.startswith("mkdir -p -- /workspace")
    # Runner-owned cleanup tears the sandbox down; stop and delete paths each make a
    # best-effort kill call and the provider tolerates repeats like upstream backends.
    assert fake.kill_calls >= 1
    assert fake.close_calls >= 1
