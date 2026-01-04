"""Tests for chat_logger.py - Chat history logging."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from openai_proxy.chat_logger import ChatLogger, get_chat_logger


class TestChatLogger:
    """Tests for ChatLogger class."""

    @pytest.fixture
    def logs_dir(self) -> Path:
        """Create a temporary logs directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def logger(self, logs_dir) -> ChatLogger:
        """Create a ChatLogger instance."""
        return ChatLogger(logs_dir)

    def test_logger_creates_directory(self):
        """Test that logger creates logs directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_path = Path(tmpdir) / "nested" / "logs"
            logger = ChatLogger(logs_path)
            
            assert logs_path.exists()

    def test_get_conversation_id_from_request(self, logger):
        """Test generating conversation ID from request."""
        request = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ]
        }
        
        conv_id = logger._get_conversation_id(request)
        
        assert conv_id is not None
        assert "test-model" in conv_id

    def test_get_conversation_id_explicit(self, logger):
        """Test using explicit conversation ID."""
        request = {
            "model": "test-model",
            "conversation_id": "my-custom-id",
            "messages": []
        }
        
        conv_id = logger._get_conversation_id(request)
        
        assert conv_id == "my-custom-id"

    def test_get_conversation_id_consistent(self, logger):
        """Test that same request generates same ID."""
        request = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Consistent message"}
            ]
        }
        
        id1 = logger._get_conversation_id(request)
        id2 = logger._get_conversation_id(request)
        
        assert id1 == id2

    def test_get_conversation_id_fallback(self, logger):
        """Test fallback ID generation when no user message."""
        request = {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are helpful."}
            ]
        }
        
        conv_id = logger._get_conversation_id(request)
        
        assert conv_id is not None
        assert "test-model" in conv_id

    def test_get_log_path(self, logger, logs_dir):
        """Test log file path generation."""
        path = logger._get_log_path("test_conversation_123")
        
        assert path.parent == logs_dir
        assert path.suffix == ".jsonl"
        assert "test_conversation_123" in str(path)

    def test_get_log_path_sanitizes_id(self, logger):
        """Test that log path sanitizes conversation ID."""
        path = logger._get_log_path("test/with:special<chars>")
        
        # Should not contain special characters in filename
        assert "/" not in path.name
        assert ":" not in path.name
        assert "<" not in path.name


class TestChatLoggerLogging:
    """Tests for logging methods."""

    @pytest.fixture
    def logs_dir(self) -> Path:
        """Create a temporary logs directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def logger(self, logs_dir) -> ChatLogger:
        """Create a ChatLogger instance."""
        return ChatLogger(logs_dir)

    def test_log_request(self, logger, logs_dir):
        """Test logging a request."""
        conv_id = "test-conv"
        request = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        }
        modified_request = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"}
            ],
        }
        
        logger.log_request(conv_id, request, modified_request)
        
        # Read the log file
        log_path = logs_dir / f"{conv_id}.jsonl"
        assert log_path.exists()
        
        with open(log_path) as f:
            entry = json.loads(f.readline())
        
        assert entry["type"] == "request"
        assert "timestamp" in entry
        assert entry["original_request"]["model"] == "gpt-4"

    def test_log_response(self, logger, logs_dir):
        """Test logging a response."""
        conv_id = "test-conv"
        response = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        
        logger.log_response(conv_id, response)
        
        log_path = logs_dir / f"{conv_id}.jsonl"
        assert log_path.exists()
        
        with open(log_path) as f:
            entry = json.loads(f.readline())
        
        assert entry["type"] == "response"
        assert entry["response"]["id"] == "chatcmpl-123"

    def test_log_tool_call(self, logger, logs_dir):
        """Test logging a tool call."""
        conv_id = "test-conv"
        
        logger.log_tool_call(
            conv_id,
            tool_name="get_time",
            arguments={"timezone": "UTC"},
            result='{"time": "12:00"}',
        )
        
        log_path = logs_dir / f"{conv_id}.jsonl"
        
        with open(log_path) as f:
            entry = json.loads(f.readline())
        
        assert entry["type"] == "tool_call"
        assert entry["tool_name"] == "get_time"
        assert entry["arguments"]["timezone"] == "UTC"
        assert entry["result"] == '{"time": "12:00"}'

    def test_log_error(self, logger, logs_dir):
        """Test logging an error."""
        conv_id = "test-conv"
        
        logger.log_error(
            conv_id,
            error="Something went wrong",
            context={"step": "processing"},
        )
        
        log_path = logs_dir / f"{conv_id}.jsonl"
        
        with open(log_path) as f:
            entry = json.loads(f.readline())
        
        assert entry["type"] == "error"
        assert entry["error"] == "Something went wrong"
        assert entry["context"]["step"] == "processing"

    def test_multiple_log_entries(self, logger, logs_dir):
        """Test multiple log entries append correctly."""
        conv_id = "test-conv"
        
        logger.log_request(conv_id, {"model": "gpt-4", "messages": []}, {"model": "gpt-4", "messages": []})
        logger.log_response(conv_id, {"id": "123", "model": "gpt-4", "choices": []})
        logger.log_tool_call(conv_id, "tool", {}, "result")
        
        log_path = logs_dir / f"{conv_id}.jsonl"
        
        with open(log_path) as f:
            lines = f.readlines()
        
        assert len(lines) == 3
        assert json.loads(lines[0])["type"] == "request"
        assert json.loads(lines[1])["type"] == "response"
        assert json.loads(lines[2])["type"] == "tool_call"


class TestChatLoggerHistory:
    """Tests for reading conversation history."""

    @pytest.fixture
    def logs_dir(self) -> Path:
        """Create a temporary logs directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def logger(self, logs_dir) -> ChatLogger:
        """Create a ChatLogger instance."""
        return ChatLogger(logs_dir)

    def test_get_conversation_history(self, logger):
        """Test reading conversation history."""
        conv_id = "test-conv"
        
        # Log some entries
        logger.log_request(conv_id, {"model": "gpt-4", "messages": []}, {"model": "gpt-4", "messages": []})
        logger.log_response(conv_id, {"id": "123", "model": "gpt-4", "choices": []})
        
        history = logger.get_conversation_history(conv_id)
        
        assert len(history) == 2
        assert history[0]["type"] == "request"
        assert history[1]["type"] == "response"

    def test_get_conversation_history_nonexistent(self, logger):
        """Test reading non-existent conversation history."""
        history = logger.get_conversation_history("nonexistent")
        
        assert history == []

    def test_get_conversation_history_empty_file(self, logger, logs_dir):
        """Test reading empty log file."""
        conv_id = "empty-conv"
        (logs_dir / f"{conv_id}.jsonl").touch()
        
        history = logger.get_conversation_history(conv_id)
        
        assert history == []


class TestGlobalChatLogger:
    """Tests for global chat logger functions."""

    def test_get_chat_logger_creates_singleton(self):
        """Test that get_chat_logger creates singleton."""
        import openai_proxy.chat_logger as chat_logger_module
        chat_logger_module._chat_logger = None
        
        with patch("openai_proxy.chat_logger.get_settings") as mock_settings:
            mock_settings.return_value.logs_dir = "/tmp/test_logs"
            
            logger1 = get_chat_logger()
            logger2 = get_chat_logger()
            
            assert logger1 is logger2

    def test_get_chat_logger_uses_settings_logs_dir(self):
        """Test that get_chat_logger uses settings logs_dir."""
        import openai_proxy.chat_logger as chat_logger_module
        chat_logger_module._chat_logger = None
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("openai_proxy.chat_logger.get_settings") as mock_settings:
                mock_settings.return_value.logs_dir = tmpdir
                
                logger = get_chat_logger()
                
                assert str(logger.logs_dir) == tmpdir


class TestChatLoggerEdgeCases:
    """Tests for edge cases in chat logging."""

    @pytest.fixture
    def logs_dir(self) -> Path:
        """Create a temporary logs directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def logger(self, logs_dir) -> ChatLogger:
        """Create a ChatLogger instance."""
        return ChatLogger(logs_dir)

    def test_log_with_unicode(self, logger, logs_dir):
        """Test logging with unicode characters."""
        conv_id = "unicode-conv"
        
        logger.log_request(
            conv_id,
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello 你好 🎉"}]},
            {"model": "gpt-4", "messages": []},
        )
        
        history = logger.get_conversation_history(conv_id)
        
        assert "你好" in str(history)
        assert "🎉" in str(history)

    def test_log_with_datetime_objects(self, logger, logs_dir):
        """Test that datetime objects are serialized correctly."""
        conv_id = "datetime-conv"
        
        # The logger should handle datetime serialization via default=str
        logger.log_error(conv_id, "Error", {"timestamp": datetime.now()})
        
        history = logger.get_conversation_history(conv_id)
        assert len(history) == 1

    def test_conversation_id_truncation(self, logger):
        """Test that very long first messages are truncated for ID generation."""
        request = {
            "model": "test",
            "messages": [
                {"role": "user", "content": "A" * 1000}
            ]
        }
        
        conv_id = logger._get_conversation_id(request)
        
        # ID should be reasonable length
        assert len(conv_id) < 200
