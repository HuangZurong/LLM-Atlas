"""
04_MCP_Tool_Server.py — Model Context Protocol Server Example

Demonstrates building an MCP server that exposes all three primitives:
- Resources: Read-only data (e.g., database schema)
- Tools: Executable actions (e.g., SQL query)
- Prompts: Reusable interaction templates (e.g., data analysis workflow)

Uses the official `mcp` Python SDK (pip install mcp).
This server communicates via stdio transport (launched as a subprocess by the host).

Reference: https://modelcontextprotocol.io/
"""

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    Prompt,
    PromptArgument,
    PromptMessage,
    GetPromptResult,
)

import json
import sqlite3
from pathlib import Path

# ---------------------------------------------------------------------------
# In-memory demo database
# ---------------------------------------------------------------------------

DB_PATH = ":memory:"

def _init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock INTEGER NOT NULL
        );
        INSERT INTO products (name, category, price, stock) VALUES
            ('Widget A', 'hardware', 29.99, 150),
            ('Widget B', 'hardware', 49.99, 80),
            ('Service Plan', 'subscription', 9.99, 9999),
            ('Premium Plan', 'subscription', 29.99, 9999);
    """)
    return conn


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

app = Server("demo-analytics-server")
db = _init_db()


# --- Resources: read-only data exposure ---

@app.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(
            uri="db://products/schema",
            name="Products Table Schema",
            description="Schema of the products table in the analytics database",
            mimeType="application/json",
        ),
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    if uri == "db://products/schema":
        cursor = db.execute("PRAGMA table_info(products)")
        columns = [
            {"name": row[1], "type": row[2], "nullable": not row[3]}
            for row in cursor.fetchall()
        ]
        return json.dumps({"table": "products", "columns": columns}, indent=2)
    raise ValueError(f"Unknown resource: {uri}")


# --- Tools: model-controlled actions ---

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="query_products",
            description=(
                "Execute a read-only SQL SELECT query against the products table. "
                "Only SELECT statements are allowed. "
                "Columns: id, name, category, price, stock."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A SELECT SQL query against the products table",
                    },
                },
                "required": ["sql"],
            },
        ),
        Tool(
            name="get_inventory_summary",
            description="Get a summary of current inventory levels grouped by category.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "query_products":
        sql = arguments.get("sql", "")
        # Safety: only allow SELECT
        if not sql.strip().upper().startswith("SELECT"):
            return [TextContent(type="text", text="Error: Only SELECT queries are allowed.")]
        try:
            cursor = db.execute(sql)
            cols = [d[0] for d in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            result = [dict(zip(cols, row)) for row in rows]
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"SQL Error: {e}")]

    elif name == "get_inventory_summary":
        cursor = db.execute(
            "SELECT category, COUNT(*) as count, SUM(stock) as total_stock, "
            "ROUND(AVG(price), 2) as avg_price FROM products GROUP BY category"
        )
        cols = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        result = [dict(zip(cols, row)) for row in rows]
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


# --- Prompts: user-controlled interaction templates ---

@app.list_prompts()
async def list_prompts() -> list[Prompt]:
    return [
        Prompt(
            name="analyze_category",
            description="Analyze products in a specific category with pricing and stock insights",
            arguments=[
                PromptArgument(
                    name="category",
                    description="Product category to analyze (e.g., 'hardware', 'subscription')",
                    required=True,
                ),
            ],
        ),
    ]


@app.get_prompt()
async def get_prompt(name: str, arguments: dict | None) -> GetPromptResult:
    if name == "analyze_category":
        category = (arguments or {}).get("category", "hardware")
        return GetPromptResult(
            description=f"Analysis of {category} products",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"Analyze the '{category}' category in our product database. "
                            "Use the query_products tool to fetch the data, then provide:\n"
                            "1. Price range and average\n"
                            "2. Stock levels and reorder recommendations\n"
                            "3. Competitive positioning suggestions"
                        ),
                    ),
                ),
            ],
        )
    raise ValueError(f"Unknown prompt: {name}")


# ---------------------------------------------------------------------------
# Entry point — stdio transport
# ---------------------------------------------------------------------------

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
