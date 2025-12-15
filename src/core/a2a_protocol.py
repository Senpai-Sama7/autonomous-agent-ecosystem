"""
A2A (Agent-to-Agent) Communication Protocol
Enterprise-grade implementation for direct agent communication and collaboration.
Based on Google's Agent-to-Agent protocol specifications.
"""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union
from enum import Enum
from datetime import datetime
import hashlib
import aiohttp

logger = logging.getLogger(__name__)


class A2AMessageType(Enum):
    """A2A message types"""

    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    CAPABILITY_QUERY = "capability_query"
    CAPABILITY_RESPONSE = "capability_response"
    STATUS_UPDATE = "status_update"
    COLLABORATION_REQUEST = "collaboration_request"
    COLLABORATION_ACCEPT = "collaboration_accept"
    COLLABORATION_REJECT = "collaboration_reject"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


class A2ATaskState(Enum):
    """Task states in A2A protocol"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DELEGATED = "delegated"


@dataclass
class AgentCard:
    """Agent identity and capability card (A2A spec)"""

    agent_id: str
    name: str
    description: str
    version: str
    capabilities: List[str]
    skills: List[Dict[str, Any]]
    endpoint: Optional[str] = None
    auth_schemes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agentId": self.agent_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "capabilities": self.capabilities,
            "skills": self.skills,
            "endpoint": self.endpoint,
            "authSchemes": self.auth_schemes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCard":
        return cls(
            agent_id=data["agentId"],
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            capabilities=data.get("capabilities", []),
            skills=data.get("skills", []),
            endpoint=data.get("endpoint"),
            auth_schemes=data.get("authSchemes", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class A2AMessage:
    """A2A protocol message"""

    message_id: str
    message_type: A2AMessageType
    sender_id: str
    recipient_id: str
    payload: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "messageId": self.message_id,
            "messageType": self.message_type.value,
            "senderId": self.sender_id,
            "recipientId": self.recipient_id,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "correlationId": self.correlation_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2AMessage":
        return cls(
            message_id=data["messageId"],
            message_type=A2AMessageType(data["messageType"]),
            sender_id=data["senderId"],
            recipient_id=data["recipientId"],
            payload=data["payload"],
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            correlation_id=data.get("correlationId"),
        )


@dataclass
class A2ATask:
    """Task shared between agents"""

    task_id: str
    name: str
    description: str
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    state: A2ATaskState = A2ATaskState.PENDING
    owner_agent_id: str = ""
    assigned_agent_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    priority: int = 5
    dependencies: List[str] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "taskId": self.task_id,
            "name": self.name,
            "description": self.description,
            "inputData": self.input_data,
            "outputData": self.output_data,
            "state": self.state.value,
            "ownerAgentId": self.owner_agent_id,
            "assignedAgentId": self.assigned_agent_id,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "artifacts": self.artifacts,
        }


class MessageBus:
    """Async message bus for A2A communication"""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the message bus"""
        self._running = True
        self._processor_task = asyncio.create_task(self._process_messages())
        logger.info("A2A Message Bus started")

    async def stop(self):
        """Stop the message bus"""
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        logger.info("A2A Message Bus stopped")

    def subscribe(self, agent_id: str, handler: Callable):
        """Subscribe an agent to receive messages"""
        if agent_id not in self._subscribers:
            self._subscribers[agent_id] = []
        self._subscribers[agent_id].append(handler)

    def unsubscribe(self, agent_id: str):
        """Unsubscribe an agent"""
        if agent_id in self._subscribers:
            del self._subscribers[agent_id]

    async def publish(self, message: A2AMessage):
        """Publish a message to the bus"""
        await self._message_queue.put(message)

    async def _process_messages(self):
        """Process messages from queue"""
        while self._running:
            try:
                message = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                await self._deliver_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Message processing error: {e}")

    async def _deliver_message(self, message: A2AMessage):
        """Deliver message to recipient"""
        recipient_id = message.recipient_id

        # Broadcast if recipient is "*"
        if recipient_id == "*":
            for handlers in self._subscribers.values():
                for handler in handlers:
                    try:
                        await handler(message)
                    except Exception as e:
                        logger.error(f"Handler error: {e}")
        elif recipient_id in self._subscribers:
            for handler in self._subscribers[recipient_id]:
                try:
                    await handler(message)
                except Exception as e:
                    logger.error(f"Handler error: {e}")


class A2AAgent(ABC):
    """Base class for A2A-capable agents"""

    def __init__(self, card: AgentCard, message_bus: MessageBus):
        self.card = card
        self.message_bus = message_bus
        self._pending_tasks: Dict[str, A2ATask] = {}
        self._collaborators: Set[str] = set()

        # Register with message bus
        message_bus.subscribe(card.agent_id, self._handle_message)

    async def _handle_message(self, message: A2AMessage):
        """Route incoming messages to handlers"""
        handlers = {
            A2AMessageType.TASK_REQUEST: self._handle_task_request,
            A2AMessageType.TASK_RESPONSE: self._handle_task_response,
            A2AMessageType.CAPABILITY_QUERY: self._handle_capability_query,
            A2AMessageType.COLLABORATION_REQUEST: self._handle_collaboration_request,
            A2AMessageType.STATUS_UPDATE: self._handle_status_update,
            A2AMessageType.HEARTBEAT: self._handle_heartbeat,
        }

        handler = handlers.get(message.message_type)
        if handler:
            await handler(message)
        else:
            logger.warning(f"Unhandled message type: {message.message_type}")

    async def send_message(
        self,
        recipient_id: str,
        msg_type: A2AMessageType,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
    ):
        """Send a message to another agent"""
        message = A2AMessage(
            message_id=str(uuid.uuid4()),
            message_type=msg_type,
            sender_id=self.card.agent_id,
            recipient_id=recipient_id,
            payload=payload,
            correlation_id=correlation_id,
        )
        await self.message_bus.publish(message)

    async def request_task(self, recipient_id: str, task: A2ATask) -> str:
        """Request another agent to perform a task"""
        task.owner_agent_id = self.card.agent_id
        self._pending_tasks[task.task_id] = task

        await self.send_message(
            recipient_id, A2AMessageType.TASK_REQUEST, {"task": task.to_dict()}
        )
        return task.task_id

    async def query_capabilities(self, recipient_id: str):
        """Query another agent's capabilities"""
        await self.send_message(recipient_id, A2AMessageType.CAPABILITY_QUERY, {})

    async def request_collaboration(
        self, recipient_id: str, task_id: str, role: str, context: Dict[str, Any]
    ):
        """Request collaboration on a task"""
        await self.send_message(
            recipient_id,
            A2AMessageType.COLLABORATION_REQUEST,
            {"taskId": task_id, "role": role, "context": context},
        )

    async def _handle_task_request(self, message: A2AMessage):
        """Handle incoming task request"""
        task_data = message.payload.get("task", {})
        task = A2ATask(
            task_id=task_data["taskId"],
            name=task_data["name"],
            description=task_data["description"],
            input_data=task_data["inputData"],
            owner_agent_id=message.sender_id,
        )

        # Check if we can handle this task
        if await self.can_handle_task(task):
            task.assigned_agent_id = self.card.agent_id
            task.state = A2ATaskState.IN_PROGRESS

            # Execute task
            try:
                result = await self.execute_task(task)
                task.output_data = result
                task.state = A2ATaskState.COMPLETED
            except Exception as e:
                task.state = A2ATaskState.FAILED
                task.output_data = {"error": str(e)}

            # Send response
            await self.send_message(
                message.sender_id,
                A2AMessageType.TASK_RESPONSE,
                {"task": task.to_dict()},
                correlation_id=message.message_id,
            )
        else:
            # Reject task
            await self.send_message(
                message.sender_id,
                A2AMessageType.ERROR,
                {"error": "Cannot handle task", "taskId": task.task_id},
                correlation_id=message.message_id,
            )

    async def _handle_task_response(self, message: A2AMessage):
        """Handle task response"""
        task_data = message.payload.get("task", {})
        task_id = task_data.get("taskId")

        if task_id in self._pending_tasks:
            task = self._pending_tasks[task_id]
            task.output_data = task_data.get("outputData")
            task.state = A2ATaskState(task_data.get("state", "completed"))
            await self.on_task_completed(task)

    async def _handle_capability_query(self, message: A2AMessage):
        """Handle capability query"""
        await self.send_message(
            message.sender_id,
            A2AMessageType.CAPABILITY_RESPONSE,
            {"card": self.card.to_dict()},
            correlation_id=message.message_id,
        )

    async def _handle_collaboration_request(self, message: A2AMessage):
        """Handle collaboration request"""
        if await self.accept_collaboration(message.payload):
            self._collaborators.add(message.sender_id)
            await self.send_message(
                message.sender_id,
                A2AMessageType.COLLABORATION_ACCEPT,
                {"taskId": message.payload.get("taskId")},
                correlation_id=message.message_id,
            )
        else:
            await self.send_message(
                message.sender_id,
                A2AMessageType.COLLABORATION_REJECT,
                {"taskId": message.payload.get("taskId")},
                correlation_id=message.message_id,
            )

    async def _handle_status_update(self, message: A2AMessage):
        """Handle status update"""
        pass  # Override in subclass

    async def _handle_heartbeat(self, message: A2AMessage):
        """Handle heartbeat"""
        pass  # Override in subclass

    @abstractmethod
    async def can_handle_task(self, task: A2ATask) -> bool:
        """Check if agent can handle a task"""
        pass

    @abstractmethod
    async def execute_task(self, task: A2ATask) -> Dict[str, Any]:
        """Execute a task"""
        pass

    async def on_task_completed(self, task: A2ATask):
        """Called when a delegated task completes"""
        pass

    async def accept_collaboration(self, request: Dict[str, Any]) -> bool:
        """Decide whether to accept collaboration"""
        return True


class A2ACoordinator:
    """Coordinates A2A communication between agents"""

    def __init__(self):
        self.message_bus = MessageBus()
        self._agents: Dict[str, A2AAgent] = {}
        self._agent_cards: Dict[str, AgentCard] = {}
        self._task_registry: Dict[str, A2ATask] = {}

    async def start(self):
        """Start the coordinator"""
        await self.message_bus.start()
        logger.info("A2A Coordinator started")

    async def stop(self):
        """Stop the coordinator"""
        await self.message_bus.stop()
        logger.info("A2A Coordinator stopped")

    def register_agent(self, agent: A2AAgent):
        """Register an agent with the coordinator"""
        self._agents[agent.card.agent_id] = agent
        self._agent_cards[agent.card.agent_id] = agent.card
        logger.info(f"Registered A2A agent: {agent.card.name}")

    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self._agents:
            self.message_bus.unsubscribe(agent_id)
            del self._agents[agent_id]
            del self._agent_cards[agent_id]

    def find_capable_agent(
        self, required_capabilities: List[str]
    ) -> Optional[AgentCard]:
        """Find an agent with required capabilities"""
        for card in self._agent_cards.values():
            if all(cap in card.capabilities for cap in required_capabilities):
                return card
        return None

    def get_all_capabilities(self) -> Dict[str, List[str]]:
        """Get all capabilities across agents"""
        return {
            agent_id: card.capabilities for agent_id, card in self._agent_cards.items()
        }

    async def broadcast_task(self, task: A2ATask) -> Optional[str]:
        """Broadcast a task to find a capable agent"""
        self._task_registry[task.task_id] = task

        # Find capable agent
        for agent_id, agent in self._agents.items():
            if await agent.can_handle_task(task):
                return agent_id
        return None


# Singleton coordinator instance
_coordinator: Optional[A2ACoordinator] = None


def get_a2a_coordinator() -> A2ACoordinator:
    """Get or create the A2A coordinator singleton"""
    global _coordinator
    if _coordinator is None:
        _coordinator = A2ACoordinator()
    return _coordinator
