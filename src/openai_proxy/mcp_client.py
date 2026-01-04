"""MCP (Model Context Protocol) server configuration and client.

This module provides a simplified MCP client that connects to servers on-demand
and manages connections properly using background tasks.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .config import get_settings

logger = logging.getLogger(__name__)


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server."""

    type: str = Field(description="Server type: 'stdio', 'sse', or 'streamablehttp'")
    
    # For stdio servers
    command: str | None = Field(default=None, description="Command to run for stdio servers")
    args: list[str] = Field(default_factory=list, description="Arguments for the command")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    
    # For SSE and streamablehttp servers
    url: str | None = Field(default=None, description="URL for SSE/HTTP servers")
    headers: dict[str, str] = Field(default_factory=dict, description="HTTP headers for authentication")
    
    # Metadata
    description: str = Field(default="", description="Description of the server")


class MCPServersConfig(BaseModel):
    """Configuration for all MCP servers."""

    servers: dict[str, MCPServerConfig] = Field(default_factory=dict)


class MCPTool(BaseModel):
    """A tool exposed by an MCP server."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    server_name: str  # Which MCP server this tool belongs to


class MCPConnection:
    """Manages a single MCP server connection."""
    
    def __init__(self, name: str, config: MCPServerConfig):
        self.name = name
        self.config = config
        self.session = None
        self.tools: dict[str, MCPTool] = {}
        self._task: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._shutdown = asyncio.Event()
        self._read_stream = None
        self._write_stream = None

    async def start(self) -> bool:
        """Start the connection in a background task."""
        self._task = asyncio.create_task(self._run())
        try:
            # Wait for connection to be ready (with timeout)
            await asyncio.wait_for(self._ready.wait(), timeout=30.0)
            return True
        except asyncio.TimeoutError:
            logger.error(f"Timeout connecting to MCP server '{self.name}'")
            await self.stop()
            return False
        except Exception as e:
            logger.error(f"Error starting MCP server '{self.name}': {e}")
            await self.stop()
            return False

    async def stop(self) -> None:
        """Stop the connection."""
        self._shutdown.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        """Run the MCP connection in the background."""
        try:
            if self.config.type == "stdio":
                await self._run_stdio()
            elif self.config.type == "sse":
                await self._run_sse()
            elif self.config.type == "streamablehttp":
                await self._run_streamablehttp()
            else:
                logger.error(f"Unknown MCP server type: {self.config.type}")
        except asyncio.CancelledError:
            logger.debug(f"MCP connection '{self.name}' cancelled")
        except Exception as e:
            logger.exception(f"MCP connection '{self.name}' error: {e}")

    async def _run_stdio(self) -> None:
        """Run a stdio MCP connection."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        if not self.config.command:
            logger.warning(f"No command specified for stdio server '{self.name}'")
            return

        # Expand environment variables
        env = {k: os.path.expandvars(v) for k, v in self.config.env.items()}
        
        server_params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=env if env else None,
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self.session = session
                
                # Discover tools
                await self._discover_tools()
                
                # Signal that we're ready
                self._ready.set()
                logger.info(f"MCP server '{self.name}' connected with {len(self.tools)} tools")
                
                # Wait for shutdown
                await self._shutdown.wait()

    async def _run_sse(self) -> None:
        """Run an SSE MCP connection."""
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        if not self.config.url:
            logger.warning(f"No URL specified for SSE server '{self.name}'")
            return

        # Expand environment variables in headers
        headers = {k: os.path.expandvars(v) for k, v in self.config.headers.items()} if self.config.headers else None

        async with sse_client(self.config.url, headers=headers) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self.session = session
                
                # Discover tools
                await self._discover_tools()
                
                # Signal that we're ready
                self._ready.set()
                logger.info(f"MCP server '{self.name}' connected with {len(self.tools)} tools")
                
                # Wait for shutdown
                await self._shutdown.wait()

    async def _run_streamablehttp(self) -> None:
        """Run a Streamable HTTP MCP connection (used by Home Assistant)."""
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        if not self.config.url:
            logger.warning(f"No URL specified for streamablehttp server '{self.name}'")
            return

        # Expand environment variables in headers
        headers = {k: os.path.expandvars(v) for k, v in self.config.headers.items()} if self.config.headers else None

        async with streamablehttp_client(self.config.url, headers=headers) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self.session = session
                
                # Discover tools
                await self._discover_tools()
                
                # Signal that we're ready
                self._ready.set()
                logger.info(f"MCP server '{self.name}' connected with {len(self.tools)} tools")
                
                # Wait for shutdown
                await self._shutdown.wait()

    async def _discover_tools(self) -> None:
        """Discover tools from the MCP server."""
        if not self.session:
            return
            
        try:
            result = await self.session.list_tools()
            for tool in result.tools:
                full_name = f"mcp_{self.name}_{tool.name}"
                self.tools[full_name] = MCPTool(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                    server_name=self.name,
                )
            logger.debug(f"Discovered {len(self.tools)} tools from '{self.name}'")
        except Exception as e:
            logger.error(f"Error discovering tools from '{self.name}': {e}")

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on this MCP server."""
        if not self.session:
            return {"error": f"MCP server '{self.name}' not connected"}

        try:
            result = await self.session.call_tool(tool_name, arguments)
            
            # Extract content from the result
            if hasattr(result, 'content') and result.content:
                contents = []
                for item in result.content:
                    if hasattr(item, 'text'):
                        contents.append(item.text)
                    elif hasattr(item, 'data'):
                        contents.append(str(item.data))
                return "\n".join(contents) if contents else str(result)
            return str(result)
            
        except Exception as e:
            logger.error(f"Error calling tool '{tool_name}' on '{self.name}': {e}")
            return {"error": f"Tool execution failed: {str(e)}"}


class MCPClient:
    """Client for managing MCP server connections."""

    def __init__(self):
        self._connections: dict[str, MCPConnection] = {}
        self._server_configs: dict[str, MCPServerConfig] = {}

    async def load_config(self, config_path: str | Path) -> None:
        """Load MCP server configurations from a YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            logger.debug(f"MCP config not found: {config_path}")
            return

        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Handle empty or missing servers key
        if not data.get("servers"):
            data["servers"] = {}

        config = MCPServersConfig.model_validate(data)
        self._server_configs = config.servers
        logger.info(f"Loaded {len(self._server_configs)} MCP server configs")

    async def connect_server(self, name: str) -> bool:
        """Connect to a specific MCP server."""
        if name in self._connections:
            return True  # Already connected

        config = self._server_configs.get(name)
        if config is None:
            logger.warning(f"MCP server '{name}' not found in configuration")
            return False

        connection = MCPConnection(name, config)
        success = await connection.start()
        
        if success:
            self._connections[name] = connection
            
        return success

    async def connect_all(self) -> None:
        """Connect to all configured MCP servers."""
        for name in self._server_configs:
            await self.connect_server(name)

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for name, connection in list(self._connections.items()):
            try:
                await connection.stop()
            except Exception as e:
                logger.error(f"Error disconnecting from MCP server '{name}': {e}")
        self._connections.clear()

    def get_tools(self, server_names: list[str] | None = None) -> list[MCPTool]:
        """Get tools from specified servers, or all tools if no servers specified."""
        tools = []
        for name, connection in self._connections.items():
            if server_names is None or name in server_names:
                tools.extend(connection.tools.values())
        return tools

    def get_tool(self, full_name: str) -> MCPTool | None:
        """Get a specific tool by its full name."""
        for connection in self._connections.values():
            if full_name in connection.tools:
                return connection.tools[full_name]
        return None

    async def call_tool(self, full_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on an MCP server."""
        # Find which connection has this tool
        for connection in self._connections.values():
            if full_name in connection.tools:
                tool = connection.tools[full_name]
                return await connection.call_tool(tool.name, arguments)
        
        return {"error": f"Tool '{full_name}' not found"}

    def get_available_servers(self) -> list[str]:
        """Get list of configured server names."""
        return list(self._server_configs.keys())

    def get_connected_servers(self) -> list[str]:
        """Get list of connected server names."""
        return list(self._connections.keys())

    def get_server_info(self, name: str) -> dict[str, Any] | None:
        """Get detailed information about a server.
        
        Args:
            name: Server name
            
        Returns:
            Server info dict or None if not found
        """
        config = self._server_configs.get(name)
        if config is None:
            return None

        connection = self._connections.get(name)
        tools = list(connection.tools.values()) if connection else []

        return {
            "name": name,
            "type": config.type,
            "description": config.description,
            "connected": name in self._connections,
            "tools_count": len(tools),
            "config": {
                "command": config.command,
                "args": config.args,
                "url": config.url,
            },
        }

    def get_all_servers_info(self) -> list[dict[str, Any]]:
        """Get information about all configured servers.
        
        Returns:
            List of server info dicts
        """
        return [
            self.get_server_info(name)
            for name in self._server_configs
            if self.get_server_info(name) is not None
        ]

    def get_server_tools(self, name: str) -> list[dict[str, Any]]:
        """Get all tools for a specific server.
        
        Args:
            name: Server name
            
        Returns:
            List of tool info dicts
        """
        connection = self._connections.get(name)
        if connection is None:
            return []

        return [
            {
                "name": tool.name,
                "full_name": full_name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
            for full_name, tool in connection.tools.items()
        ]


# Global MCP client instance
_mcp_client: MCPClient | None = None


async def get_mcp_client() -> MCPClient:
    """Get the global MCP client, initializing if needed."""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPClient()
        settings = get_settings()
        config_path = Path(settings.mcp_config_dir) / "servers.yaml"
        await _mcp_client.load_config(config_path)
    return _mcp_client


async def initialize_mcp_client() -> MCPClient:
    """Initialize and connect the MCP client."""
    client = await get_mcp_client()
    await client.connect_all()
    return client


async def shutdown_mcp_client() -> None:
    """Shutdown the MCP client."""
    global _mcp_client
    if _mcp_client is not None:
        await _mcp_client.disconnect_all()
        _mcp_client = None
