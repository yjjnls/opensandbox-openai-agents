"""Complete scripted demo: one agent turn executing inside an OpenSandbox sandbox.

The model is scripted, so no model API key is needed -- only a reachable OpenSandbox
endpoint and credentials. Set the environment variables before running:

    export OPENSANDBOX_DOMAIN="sandbox.example.com"
    export OPENSANDBOX_API_KEY="<access_key_id>:<secret_key>"
    export OPENSANDBOX_IMAGE="opensandbox/code-interpreter:latest"  # optional

Then: python demo/demo.py
"""

from __future__ import annotations

import asyncio
import os
from datetime import timedelta

from agents import RunConfig, Runner
from agents.sandbox import SandboxRunConfig
from opensandbox.config import ConnectionConfig
from opensandbox_agents import OpenSandboxClient, OpenSandboxClientOptions


def build_client() -> OpenSandboxClient:
    domain = os.environ["OPENSANDBOX_DOMAIN"]
    api_key = os.environ["OPENSANDBOX_API_KEY"]
    connection_config = ConnectionConfig(
        domain=domain,
        api_key=api_key,
        protocol=os.environ.get("OPENSANDBOX_PROTOCOL", "https"),
        request_timeout=timedelta(seconds=60),
    )
    return OpenSandboxClient(connection_config=connection_config)


async def main() -> None:
    client = build_client()
    image = os.environ.get("OPENSANDBOX_IMAGE", "opensandbox/code-interpreter:latest")
    run_config = RunConfig(
        sandbox=SandboxRunConfig(
            client=client,
            options=OpenSandboxClientOptions(
                image=image,
                workspace_root="/workspace",
                sandbox_ttl_s=1800,
                resource={"cpu": "1", "memory": "2Gi"},
            ),
        ),
    )

    from agents import Agent

    agent = Agent(name="sandboxed-demo", instructions="Use the sandbox shell when asked.")

    try:
        result = await Runner.run(
            agent,
            "Run `uname -a` in the sandbox and summarize the output.",
            run_config=run_config,
        )
        print(result.final_output)
    finally:
        # The runner tears down the session it created. Closing the client releases
        # client-held resources; running sandboxes are stopped through their sessions.
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
