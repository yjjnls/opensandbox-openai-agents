# opensandbox-openai-agents

[OpenSandbox](https://github.com/opensandbox-group/OpenSandbox) sandbox provider for the
[OpenAI Agents SDK](https://github.com/openai/openai-agents-python) (`openai-agents`).

This package implements the Agents SDK sandbox client/session interfaces on top of the
OpenSandbox Python SDK (`opensandbox`), so agents can run their shell tool and workspace
I/O inside a remote OpenSandbox sandbox. It follows the SDK's
[external sandbox provider guide](https://github.com/openai/openai-agents-python/blob/main/docs/sandbox/custom-providers.md):
the provider ships in its own distribution, and applications pass the client to
`SandboxRunConfig(client=...)` explicitly. The Agents SDK does not need to import this
package or register any entry points.

## Install

```bash
pip install opensandbox-openai-agents
```

Pinned compatibility ranges (tested together):

- `openai-agents>=0.24.0,<0.25.0`
- `opensandbox>=1.2.0,<2.0.0`

## Usage

```python
from agents import RunConfig, Runner
from agents.sandbox import SandboxRunConfig
from opensandbox.config import ConnectionConfig
from opensandbox_agents import OpenSandboxClient, OpenSandboxClientOptions

client = OpenSandboxClient(
    connection_config=ConnectionConfig(
        domain="sandbox.example.com",       # your OpenSandbox API endpoint
        api_key="<access_key_id>:<secret_key>",
        protocol="https",
    ),
)

run_config = RunConfig(
    sandbox=SandboxRunConfig(
        client=client,
        options=OpenSandboxClientOptions(
            image="opensandbox/code-interpreter:latest",
            sandbox_ttl_s=3600,              # optional sandbox TTL
            resource={"cpu": "1", "memory": "2Gi"},
            env_vars={"MY_VAR": "value"},
        ),
    ),
)

result = await Runner.run(agent, "inspect the workspace", run_config=run_config)
```

The connection configuration (endpoint and credentials) lives on the client and is never
serialized. Options are serializable and round-trip through persisted session state; keep
them free of secrets.

### Options

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `image` | `str \| None` | — | Sandbox image. Exactly one of `image`/`snapshot_id` is required. |
| `snapshot_id` | `str \| None` | — | Create from an OpenSandbox image snapshot instead of a raw image. |
| `workspace_root` | `str` | `/workspace` | Workspace root inside the sandbox (agent cwd). |
| `env_vars` | `dict[str, str] \| None` | — | Environment variables injected at creation. |
| `metadata` | `dict[str, str] \| None` | — | Provider metadata labels. |
| `resource` | `dict[str, str] \| None` | — | Resource spec, e.g. `{"cpu": "1", "memory": "2Gi"}`. |
| `entrypoint` | `tuple[str, ...] \| None` | — | Container entrypoint override. |
| `sandbox_ttl_s` | `int \| None` | — | Sandbox time-to-live in seconds. |
| `ready_timeout_s` | `int` | `60` | Readiness wait when creating. |
| `pause_on_exit` | `bool` | `False` | Pause instead of delete when the session shuts down. |
| `timeouts` | `OpenSandboxTimeouts \| dict` | — | Per-operation timeout overrides (seconds). |
| `exposed_ports` | `tuple[int, ...]` | `()` | Ports resolvable via the exposed-port endpoint API. |

### Resume behavior

Persisted state carries the sandbox id. On resume the provider reconnects to the existing
sandbox; if it is unreachable it recreates one from the recorded creation inputs and
restores the workspace from the SDK snapshot. When the session exits with
`pause_on_exit=True`, resume first tries the OpenSandbox pause/resume path.

## Capability notes and limitations

Implemented on top of OpenSandbox execd (commands + files APIs):

- Shell command execution with wall-clock timeout mapping, per-command environment, and
  workspace-root cwd (`ExecTimeoutError`/`ExecTransportError` with retryability metadata).
- File read/write within the workspace, with path-confinement validation against the
  workspace root.
- Workspace persistence: tar snapshot/restore over the execd file API, including the
  Agents SDK snapshot/resume semantics.
- Lifecycle: create, resume/recreate, delete (kill or pause), TTL, metadata, resources.
- Exposed ports via `get_endpoint` URL mapping.

Not supported in this version (rejected explicitly, never silently):

- PTY / interactive stdin sessions (`supports_pty()` is `False`; the SDK shell tool
  automatically uses one-shot exec instead).
- Storage mounts (`Mount` manifest entries raise `MountConfigError`).
- Sandbox users mapping (`uid`/`gid` execution) beyond the default account.

The provider image must run OpenSandbox execd; official `opensandbox/*` images do.

## Development

```bash
python -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
./.venv/bin/pytest
```

Tests are mock-based and require no sandbox service or model API key. `demo/demo.py`
shows a complete scripted run against a real OpenSandbox endpoint.

## License

Apache-2.0. This is an independent integration package and is not affiliated with or
endorsed by OpenAI or the OpenSandbox project.
