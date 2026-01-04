"""Chat history logging for the OpenAI API Proxy."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import get_settings

logger = logging.getLogger(__name__)


class ChatLogger:
    """Logger for storing complete chat histories per conversation."""

    def __init__(self, logs_dir: str | Path = "logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._conversations: dict[str, list[dict[str, Any]]] = {}

    def _get_conversation_id(self, request: dict[str, Any]) -> str:
        """Get or generate a conversation ID for the request.
        
        Uses the first message's hash + model as an identifier, or generates
        a timestamp-based ID if no consistent identifier can be derived.
        """
        # Try to use a conversation ID if provided in the request
        if "conversation_id" in request:
            return request["conversation_id"]
        
        # Use model + first user message as conversation key
        model = request.get("model", "unknown")
        messages = request.get("messages", [])
        
        # Find first user message to use as conversation anchor
        first_user_msg = None
        for msg in messages:
            if msg.get("role") == "user":
                first_user_msg = msg.get("content", "")[:100]
                break
        
        if first_user_msg:
            # Create a hash-based ID from model + first message
            import hashlib
            key = f"{model}:{first_user_msg}"
            conv_id = hashlib.sha256(key.encode()).hexdigest()[:16]
            return f"{model}_{conv_id}"
        
        # Fallback to timestamp-based ID
        return f"{model}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    def _get_log_path(self, conversation_id: str) -> Path:
        """Get the log file path for a conversation."""
        # Sanitize the conversation ID for use as filename
        safe_id = "".join(c if c.isalnum() or c in "_-" else "_" for c in conversation_id)
        return self.logs_dir / f"{safe_id}.jsonl"

    def log_request(
        self,
        conversation_id: str,
        request: dict[str, Any],
        modified_request: dict[str, Any],
    ) -> None:
        """Log a request to the conversation history."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "request",
            "original_request": {
                "model": request.get("model"),
                "messages": request.get("messages"),
                "tools": request.get("tools"),
                "stream": request.get("stream"),
            },
            "modified_request": {
                "model": modified_request.get("model"),
                "messages": modified_request.get("messages"),
                "tools": modified_request.get("tools"),
            },
        }
        
        self._append_log(conversation_id, log_entry)

    def log_response(
        self,
        conversation_id: str,
        response: dict[str, Any],
    ) -> None:
        """Log a response to the conversation history."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "response",
            "response": {
                "id": response.get("id"),
                "model": response.get("model"),
                "choices": response.get("choices"),
                "usage": response.get("usage"),
            },
        }
        
        self._append_log(conversation_id, log_entry)

    def log_tool_call(
        self,
        conversation_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: str,
    ) -> None:
        """Log a tool call execution."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "tool_call",
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
        }
        
        self._append_log(conversation_id, log_entry)

    def log_error(
        self,
        conversation_id: str,
        error: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Log an error in the conversation."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "error": error,
            "context": context,
        }
        
        self._append_log(conversation_id, log_entry)

    def _append_log(self, conversation_id: str, entry: dict[str, Any]) -> None:
        """Append a log entry to the conversation file."""
        log_path = self._get_log_path(conversation_id)
        
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            logger.debug(f"Logged {entry['type']} to {log_path}")
        except Exception as e:
            logger.error(f"Failed to write log: {e}")

    def get_conversation_history(self, conversation_id: str) -> list[dict[str, Any]]:
        """Read the full conversation history."""
        log_path = self._get_log_path(conversation_id)
        
        if not log_path.exists():
            return []
        
        entries = []
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        
        return entries


# Global chat logger instance
_chat_logger: ChatLogger | None = None


def get_chat_logger() -> ChatLogger:
    """Get the global chat logger instance."""
    global _chat_logger
    if _chat_logger is None:
        settings = get_settings()
        logs_dir = Path(settings.logs_dir)
        _chat_logger = ChatLogger(logs_dir)
    return _chat_logger
