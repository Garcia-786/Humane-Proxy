# Integrations

Using HumaneProxy from AI agents and other languages: the MCP server,
native framework adapters, and Node.js.

## Table of Contents

- [MCP Server (for AI Agents)](#mcp-server-for-ai-agents)
- [LlamaIndex](#llamaindex)
- [CrewAI](#crewai)
- [AutoGen (AG2)](#autogen-ag2)
- [LangChain](#langchain)
- [Node.js / TypeScript (via MCP)](#nodejs--typescript-via-mcp)

---

## MCP Server (for AI Agents)

```bash
pip install humane-proxy[mcp]
humane-proxy mcp-serve                                # stdio (default)
humane-proxy mcp-serve --transport http --port 3000   # HTTP on 127.0.0.1
```

Exposes three tools via Model Context Protocol:

| Tool | Description |
|---|---|
| `check_message_safety` | Full pipeline classification |
| `get_session_risk` | Read-only session trajectory snapshot (trend, spike, category counts) |
| `list_recent_escalations` | Bounded audit log query |

Available on the [Official MCP Registry](https://registry.modelcontextprotocol.io),
[Glama](https://glama.ai/mcp/servers/Vishisht16/Humane-Proxy), and
[MCP Marketplace](https://mcp-marketplace.io/server/io-github-vishisht16-humane-proxy).

### Claude Desktop / Cursor

Add to `claude_desktop_config.json` (or your client's MCP config):

```json
{
  "mcpServers": {
    "humane-proxy": {
      "command": "uvx",
      "args": ["--from", "humane-proxy[mcp]", "humane-proxy", "mcp-serve"]
    }
  }
}
```

### Securing HTTP MCP

HTTP MCP is local-only by default. To bind publicly, pass `--host 0.0.0.0`
explicitly and protect tool access with a bearer token:

```bash
export HUMANE_PROXY_ADMIN_KEY=your-secret-token
humane-proxy mcp-serve --transport http --host 0.0.0.0 --port 3000
```

Clients must send `Authorization: Bearer your-secret-token` when the token is
configured. Leave `HUMANE_PROXY_ADMIN_KEY` unset for stdio/local-only MCP.

---

## LlamaIndex

```bash
pip install humane-proxy[llamaindex]
```
```python
from humane_proxy.integrations.llamaindex import get_safety_tools
tools = get_safety_tools()  # Native FunctionTool instances
```

## CrewAI

```bash
pip install humane-proxy[crewai]
```
```python
from humane_proxy.integrations.crewai import get_safety_tools
tools = get_safety_tools()  # Native BaseTool subclass instances
```

## AutoGen (AG2)

```bash
pip install humane-proxy[autogen]
```
```python
from humane_proxy.integrations.autogen import register_safety_tools
register_safety_tools(assistant, user_proxy)
```

## LangChain

```bash
pip install humane-proxy[langchain]
```

```python
from humane_proxy.integrations.langchain import get_safety_tools

# Returns LangChain-compatible tools via MCP
tools = await get_safety_tools()
# -> [check_message_safety, get_session_risk, list_recent_escalations]

# Or get the config dict for MultiServerMCPClient:
from humane_proxy.integrations.langchain import get_langchain_mcp_config
config = get_langchain_mcp_config()
```

---

## Node.js / TypeScript (via MCP)

MCP is language-agnostic, so Node.js apps can use HumaneProxy today with
the official MCP SDK — no wrapper package needed. The server runs as a
child process over stdio (requires Python or [uv](https://docs.astral.sh/uv/)
on the machine).

```bash
npm install @modelcontextprotocol/sdk
```

```js
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const transport = new StdioClientTransport({
  command: "uvx",
  args: ["--from", "humane-proxy[mcp]", "humane-proxy", "mcp-serve"],
});

const client = new Client({ name: "my-app", version: "1.0.0" });
await client.connect(transport);

const result = await client.callTool({
  name: "check_message_safety",
  arguments: { message: "I want to end my life", session_id: "user-42" },
});
console.log(result.content);
// -> safety verdict, category, score, triggers
```

### With the Vercel AI SDK

The AI SDK can consume MCP tools directly, making HumaneProxy's checks
available to any model call:

```js
import { experimental_createMCPClient } from "ai";
import { Experimental_StdioMCPTransport } from "ai/mcp-stdio";

const mcpClient = await experimental_createMCPClient({
  transport: new Experimental_StdioMCPTransport({
    command: "uvx",
    args: ["--from", "humane-proxy[mcp]", "humane-proxy", "mcp-serve"],
  }),
});

const tools = await mcpClient.tools();
// Pass `tools` to generateText / streamText
```

### Alternative: the REST proxy

If you'd rather not spawn a child process, run `humane-proxy start` as a
sidecar service and point your existing OpenAI-style client at it — the
proxy forwards safe messages upstream and answers flagged ones itself.
No SDK required; it's just HTTP.
