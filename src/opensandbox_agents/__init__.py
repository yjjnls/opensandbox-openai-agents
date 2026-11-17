"""OpenSandbox sandbox provider for the OpenAI Agents SDK.

Importing this package registers the `opensandbox` client-options and session-state
types with the Agents SDK, so serialized state round-trips work in every process
that imports it (including resume workers).

Basic usage:

```python
from opensandbox.config import ConnectionConfig
from opensandbox_agents import OpenSandboxClient, OpenSandboxClientOptions

from agents import RunConfig
from agents.sandbox import SandboxRunConfig

client = OpenSandboxClient(
    connection_config=ConnectionConfig(
        domain="sandbox.example.com",
        api_key="<access_key_id>:<secret_key>",
        protocol="https",
    ),
)
run_config = RunConfig(
    sandbox=SandboxRunConfig(
        client=client,
        options=OpenSandboxClientOptions(image="opensandbox/code-interpreter:latest"),
    ),
)
# Pass run_config to Runner.run(...). Close sandboxes through the client/sessions.
```
"""

from .sandbox import (
    DEFAULT_OPENSANDBOX_WORKSPACE_ROOT,
    OpenSandboxClient,
    OpenSandboxClientOptions,
    OpenSandboxSession,
    OpenSandboxSessionState,
    OpenSandboxTimeouts,
)

__all__ = [
    "DEFAULT_OPENSANDBOX_WORKSPACE_ROOT",
    "OpenSandboxClient",
    "OpenSandboxClientOptions",
    "OpenSandboxSession",
    "OpenSandboxSessionState",
    "OpenSandboxTimeouts",
    "__version__",
]

__version__ = "0.1.0"
