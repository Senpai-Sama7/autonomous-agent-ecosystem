"""
MCP (Model Context Protocol) Integration Module
Enterprise-grade implementation for tool and resource access via MCP servers.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum
import aiohttp
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class MCPMessageType(Enum):
    """MCP message types per protocol spec"""

    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


@dataclass
class MCPTool:
    """Represents an MCP tool definition"""

    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "serverName": self.server_name,
        }


@dataclass
class MCPResource:
    """Represents an MCP resource"""

    uri: str
    name: str
    description: str
    mime_type: str = "text/plain"
    server_name: str = ""


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server"""

    name: str
    transport: str  # "stdio" | "http" | "websocket"
    command: Optional[str] = None  # For stdio transport
    url: Optional[str] = None  # For http/websocket transport
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)


class MCPTransport(ABC):
    """Abstract base for MCP transport implementations"""

    @abstractmethod
    async def connect(self) -> bool:
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        pass

    @abstractmethod
    async def send(self, message: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        pass


class HTTPTransport(MCPTransport):
    """HTTP-based MCP transport"""

    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None):
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}
        self._session: Optional[aiohttp.ClientSession] = None
        self._connected = False

    async def connect(self) -> bool:
        try:
            self._session = aiohttp.ClientSession(headers=self.headers)
            # Test connection with initialize
            response = await self.send(
                {
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "ASTRO", "version": "1.0.0"},
                    },
                    "id": 1,
                }
            )
            self._connected = "result" in response
            return self._connected
        except Exception as e:
            logger.error(f"HTTP transport connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False

    async def send(self, message: Dict[str, Any]) -> Dict[str, Any]:
        if not self._session:
            raise ConnectionError("Not connected")

        async with self._session.post(self.url, json=message) as response:
            return await response.json()

    def is_connected(self) -> bool:
        return self._connected


class MCPClient:
    """Client for communicating with MCP servers"""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.transport: Optional[MCPTransport] = None
        self._tools: Dict[str, MCPTool] = {}
        self._resources: Dict[str, MCPResource] = {}
        self._request_id = 0
        self._initialized = False

    async def connect(self) -> bool:
        """Establish connection to MCP server"""
        try:
            if self.config.transport == "http":
                self.transport = HTTPTransport(self.config.url)
            else:
                logger.warning(
                    f"Transport {self.config.transport} not fully implemented, using HTTP"
                )
                self.transport = HTTPTransport(
                    self.config.url or "http://localhost:3000"
                )

            if await self.transport.connect():
                await self._discover_capabilities()
                self._initialized = True
                logger.info(f"Connected to MCP server: {self.config.name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {self.config.name}: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from MCP server"""
        if self.transport:
            await self.transport.disconnect()
        self._initialized = False

    async def _discover_capabilities(self) -> None:
        """Discover available tools and resources"""
        # List tools
        tools_response = await self._send_request("tools/list", {})
        if "result" in tools_response:
            for tool in tools_response["result"].get("tools", []):
                mcp_tool = MCPTool(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    input_schema=tool.get("inputSchema", {}),
                    server_name=self.config.name,
                )
                self._tools[tool["name"]] = mcp_tool

        # List resources
        resources_response = await self._send_request("resources/list", {})
        if "result" in resources_response:
            for resource in resources_response["result"].get("resources", []):
                mcp_resource = MCPResource(
                    uri=resource["uri"],
                    name=resource.get("name", ""),
                    description=resource.get("description", ""),
                    mime_type=resource.get("mimeType", "text/plain"),
                    server_name=self.config.name,
                )
                self._resources[resource["uri"]] = mcp_resource

    async def _send_request(
        self, method: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send JSON-RPC request to server"""
        self._request_id += 1
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self._request_id,
        }
        return await self.transport.send(message)

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool on the MCP server"""
        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        response = await self._send_request(
            "tools/call", {"name": tool_name, "arguments": arguments}
        )

        if "error" in response:
            raise RuntimeError(f"Tool execution failed: {response['error']}")

        return response.get("result", {}).get("content", [])

    async def read_resource(self, uri: str) -> Any:
        """Read a resource from the MCP server"""
        response = await self._send_request("resources/read", {"uri": uri})

        if "error" in response:
            raise RuntimeError(f"Resource read failed: {response['error']}")

        return response.get("result", {}).get("contents", [])

    def get_tools(self) -> List[MCPTool]:
        return list(self._tools.values())

    def get_resources(self) -> List[MCPResource]:
        return list(self._resources.values())


class MCPRegistry:
    """Registry for managing multiple MCP servers"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._clients: Dict[str, MCPClient] = {}
            cls._instance._tool_index: Dict[str, str] = {}  # tool_name -> server_name
        return cls._instance

    async def register_server(self, config: MCPServerConfig) -> bool:
        """Register and connect to an MCP server"""
        client = MCPClient(config)
        if await client.connect():
            self._clients[config.name] = client
            # Index tools by name
            for tool in client.get_tools():
                self._tool_index[tool.name] = config.name
            logger.info(
                f"Registered MCP server: {config.name} with {len(client.get_tools())} tools"
            )
            return True
        return False

    async def unregister_server(self, name: str) -> None:
        """Disconnect and remove an MCP server"""
        if name in self._clients:
            await self._clients[name].disconnect()
            # Remove tool index entries
            self._tool_index = {k: v for k, v in self._tool_index.items() if v != name}
            del self._clients[name]

    def get_all_tools(self) -> List[MCPTool]:
        """Get all available tools from all servers"""
        tools = []
        for client in self._clients.values():
            tools.extend(client.get_tools())
        return tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool, automatically routing to correct server"""
        if tool_name not in self._tool_index:
            raise ValueError(f"Tool not found: {tool_name}")

        server_name = self._tool_index[tool_name]
        client = self._clients[server_name]
        return await client.call_tool(tool_name, arguments)

    def get_client(self, name: str) -> Optional[MCPClient]:
        return self._clients.get(name)

    @property
    def server_count(self) -> int:
        return len(self._clients)

    @property
    def tool_count(self) -> int:
        return len(self._tool_index)


class MCPToolExecutor:
    """Executes MCP tools with retry logic and error handling"""

    def __init__(self, registry: MCPRegistry, max_retries: int = 3):
        self.registry = registry
        self.max_retries = max_retries
        self._execution_history: List[Dict[str, Any]] = []

    async def execute(
        self, tool_name: str, arguments: Dict[str, Any], timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Execute a tool with retry logic"""
        execution_id = hashlib.md5(
            f"{tool_name}{json.dumps(arguments)}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        result = {
            "execution_id": execution_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "status": "pending",
            "result": None,
            "error": None,
            "attempts": 0,
            "timestamp": datetime.now().isoformat(),
        }

        for attempt in range(self.max_retries):
            result["attempts"] = attempt + 1
            try:
                tool_result = await asyncio.wait_for(
                    self.registry.call_tool(tool_name, arguments), timeout=timeout
                )
                result["status"] = "success"
                result["result"] = tool_result
                break
            except asyncio.TimeoutError:
                result["error"] = f"Timeout after {timeout}s"
                result["status"] = "timeout"
            except Exception as e:
                result["error"] = str(e)
                result["status"] = "error"

            if attempt < self.max_retries - 1:
                await asyncio.sleep(2**attempt)  # Exponential backoff

        self._execution_history.append(result)
        return result

    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self._execution_history[-limit:]


# Factory function for easy setup
async def create_mcp_registry(configs: List[Dict[str, Any]]) -> MCPRegistry:
    """Create and initialize MCP registry with server configs"""
    registry = MCPRegistry()

    for config_dict in configs:
        config = MCPServerConfig(
            name=config_dict["name"],
            transport=config_dict.get("transport", "http"),
            url=config_dict.get("url"),
            command=config_dict.get("command"),
            args=config_dict.get("args", []),
            env=config_dict.get("env", {}),
            capabilities=config_dict.get("capabilities", []),
        )
        await registry.register_server(config)

    return registry
