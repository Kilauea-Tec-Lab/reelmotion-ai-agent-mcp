# ReelMotion MCP Server

This is a Model Context Protocol (MCP) server for ReelMotion.
It provides tools to generate images and videos, and handle chat interactions.

## Prerequisites

- Python 3.10 or higher
- `pip`

## Installation

1. Navigate to the `reelmotion_mcp` directory.
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   This will install `fastmcp` and other required packages.

## Running the Server

To run the server in STDIO mode (for integration with MCP clients like Claude Desktop or Laravel subprocesses):

```bash
python server.py
```

Or using the `fastmcp` CLI:

```bash
fastmcp run server.py:mcp
```

## Tools

- `generate_image(prompt, style, size)`: Generates an image.
- `generate_video(prompt, duration, fps)`: Generates a video.
- `chat(message, context)`: Processes a chat message.

## Integration with Laravel

Laravel can interact with this MCP server by spawning it as a subprocess and communicating via Standard Input/Output (STDIO) using the JSON-RPC protocol defined by MCP.
