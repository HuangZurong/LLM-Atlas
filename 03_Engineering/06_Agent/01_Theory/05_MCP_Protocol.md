# Model Context Protocol (MCP)

*Prerequisite: [02_Agent_Architecture.md](02_Agent_Architecture.md).*

---

MCP is an open standard (introduced by Anthropic, 2024) that defines how AI applications discover and interact with external data sources and tools. It is the "USB-C of the AI era" — a universal interface replacing bespoke glue code.

## 1. Why MCP Exists

Before MCP, every Agent-tool integration was a custom implementation:

```
Without MCP:                          With MCP:
┌─────────┐   custom   ┌──────┐      ┌─────────┐         ┌──────────┐
│ Agent A  │──glue──────│ DB   │      │ Agent A  │──MCP──>│ DB Server│
│ Agent B  │──glue──────│ DB   │      │ Agent B  │──MCP──>│ DB Server│
│ Agent A  │──glue──────│ Slack│      │ Agent A  │──MCP──>│Slack Srvr│
│ Agent B  │──glue──────│ Slack│      │ Agent B  │──MCP──>│Slack Srvr│
└─────────┘             └──────┘      └─────────┘         └──────────┘
  N×M integrations                      N+M integrations
```

**Core Value**: Write one MCP Server for your database → every MCP-compliant agent can use it immediately.

## 2. Architecture

```
┌──────────────────────────────────────────────────────┐
│                    MCP Host                          │
│  (IDE, Desktop App, Backend Service)                 │
│                                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │ MCP Client │  │ MCP Client │  │ MCP Client │    │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘    │
└────────┼───────────────┼───────────────┼────────────┘
         │ JSON-RPC      │ JSON-RPC      │ JSON-RPC
    ┌────▼────┐    ┌─────▼─────┐   ┌─────▼─────┐
    │ GitHub  │    │ Database  │   │ Filesystem│
    │ Server  │    │ Server    │   │ Server    │
    └─────────┘    └───────────┘   └───────────┘
```

| Component        | Role                                                                     | Example                                        |
| :--------------- | :----------------------------------------------------------------------- | :--------------------------------------------- |
| **Host**   | The application that embeds the AI model                                 | Claude Desktop, VS Code, a backend service     |
| **Client** | Protocol handler inside the host; maintains 1:1 connection with a server | Built into the host SDK                        |
| **Server** | Lightweight program exposing specific capabilities via JSON-RPC          | `github-mcp-server`, `postgres-mcp-server` |

**Transport**: MCP supports two transport mechanisms:

- **stdio**: Server runs as a child process; communication via stdin/stdout. Best for local tools.
- **SSE (Server-Sent Events)**: HTTP-based; server runs remotely. Best for shared/cloud services.

## 3. The Three Primitives

MCP servers expose capabilities through exactly three primitive types. Understanding these is the key to designing effective MCP integrations.

### 3.1 Resources (Data Exposure)

Resources provide **read-only data** to the model's context. The model does not "call" a resource — the host application decides when to inject it.

```json
{
  "uri": "file:///project/src/main.py",
  "name": "main.py",
  "mimeType": "text/x-python",
  "description": "The main entry point of the application"
}
```

| Aspect               | Detail                                                                                    |
| :------------------- | :---------------------------------------------------------------------------------------- |
| **Control**    | Application-controlled (host decides when to attach)                                      |
| **Analogy**    | Like opening a file in an IDE — the user/app chooses what to show the model              |
| **Use Cases**  | File contents, database schemas, API documentation, configuration                         |
| **URI Scheme** | Custom schemes allowed (e.g.,`postgres://db/users/schema`, `github://repo/issues/42`) |

**Dynamic Resources**: Servers can expose resource templates with URI patterns:

```
github://repos/{owner}/{repo}/issues/{issue_number}
```

### 3.2 Tools (Action Execution)

Tools are **model-controlled functions** that the LLM can invoke via function calling. This is the most powerful primitive — it lets agents take actions in the real world.

```json
{
  "name": "query_database",
  "description": "Execute a read-only SQL query against the analytics database",
  "inputSchema": {
    "type": "object",
    "properties": {
      "sql": { "type": "string", "description": "The SQL query to execute" }
    },
    "required": ["sql"]
  }
}
```

| Aspect              | Detail                                                                      |
| :------------------ | :-------------------------------------------------------------------------- |
| **Control**   | Model-controlled (LLM decides when to call, with user approval)             |
| **Analogy**   | Like function calling / tool use in OpenAI or Anthropic APIs                |
| **Use Cases** | Database queries, API calls, file writes, code execution, sending messages  |
| **Safety**    | Tools should include human-in-the-loop confirmation for destructive actions |

**Critical Design Rule**: Tools must have clear, unambiguous descriptions. The LLM selects tools based on the `description` field — a vague description leads to misuse.

### 3.3 Prompts (Interaction Templates)

Prompts are **user-controlled templates** that define reusable interaction patterns. They are the least understood but most elegant primitive.

```json
{
  "name": "code_review",
  "description": "Review code for bugs, security issues, and style",
  "arguments": [
    { "name": "language", "description": "Programming language", "required": true },
    { "name": "code", "description": "The code to review", "required": true }
  ]
}
```

| Aspect              | Detail                                                                     |
| :------------------ | :------------------------------------------------------------------------- |
| **Control**   | User-controlled (user explicitly selects a prompt template)                |
| **Analogy**   | Like slash commands in Slack or IDE — user triggers a predefined workflow |
| **Use Cases** | Code review templates, report generation, analysis workflows               |
| **Output**    | Returns structured messages (system + user) that populate the conversation |

### 3.4 Primitive Selection Guide

```
Need to give the model background data?        → Resource
Need the model to take an action?               → Tool
Need a reusable user-triggered workflow?         → Prompt
```

|                             | Resources      | Tools                    | Prompts             |
| :-------------------------- | :------------- | :----------------------- | :------------------ |
| **Controlled by**     | Application    | Model (LLM)              | User                |
| **Direction**         | Data → Model  | Model → External System | User → Model       |
| **Requires approval** | No (read-only) | Yes (side effects)       | No (user-initiated) |
| **Analogous to**      | GET request    | POST request             | Slash command       |

## 4. Lifecycle & Capability Negotiation

```
Client                              Server
  │                                    │
  │──── initialize ──────────────────>│
  │<─── capabilities (tools,          │
  │      resources, prompts) ─────────│
  │                                    │
  │──── initialized ─────────────────>│
  │                                    │
  │──── tools/list ──────────────────>│  (discovery)
  │<─── tool definitions ────────────│
  │                                    │
  │──── tools/call ──────────────────>│  (execution)
  │<─── result ──────────────────────│
  │                                    │
  │──── notifications/... ───────────>│  (updates)
```

Key lifecycle features:

- **Capability Negotiation**: During `initialize`, the server declares which primitives it supports. The client adapts accordingly.
- **Dynamic Updates**: Servers can notify clients when their tool/resource list changes (e.g., new tables added to a database).
- **Stateful Sessions**: Each client-server connection maintains session state, enabling multi-turn tool interactions.

## 5. Security Model

| Layer                       | Mechanism                                | Purpose                         |
| :-------------------------- | :--------------------------------------- | :------------------------------ |
| **Transport**         | TLS for SSE; process isolation for stdio | Encrypt data in transit         |
| **Authentication**    | OAuth 2.0 (for remote servers)           | Verify server identity          |
| **Authorization**     | Per-tool permission scoping              | Limit what the model can do     |
| **Human-in-the-Loop** | Approval prompts for destructive tools   | Prevent unintended side effects |
| **Input Validation**  | JSON Schema on tool inputs               | Reject malformed requests       |
| **Rate Limiting**     | Server-side throttling                   | Prevent abuse / runaway agents  |

**The Principle of Least Privilege**: An MCP server should expose the minimum set of tools needed. A database server for analytics should expose `read_query` but NOT `write_query` unless explicitly required.

## 6. MCP vs. Alternatives

|                       | MCP                     | OpenAI Function Calling | LangChain Tools    | Custom REST API |
| :-------------------- | :---------------------- | :---------------------- | :----------------- | :-------------- |
| **Standard**    | Open protocol           | Proprietary             | Framework-specific | Ad-hoc          |
| **Discovery**   | Dynamic (tools/list)    | Static (in prompt)      | Static (in code)   | Manual          |
| **Transport**   | stdio / SSE             | HTTP                    | In-process         | HTTP            |
| **Multi-model** | Any MCP host            | OpenAI only             | Any (via adapter)  | Any             |
| **Ecosystem**   | Growing (1000+ servers) | N/A                     | Large              | N/A             |

## Key References

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [Anthropic MCP Documentation](https://modelcontextprotocol.io/)
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)
