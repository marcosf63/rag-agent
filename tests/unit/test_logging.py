"""Tests for JSON logging functionality."""

import json
import logging
import sys
from pathlib import Path
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rag_agent.utils.logging import JsonFormatter, setup_logger


class TestJsonFormatter:
    """Tests for JsonFormatter class."""
    
    def test_basic_formatting(self):
        """Test basic JSON log formatting."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        result = formatter.format(record)
        parsed = json.loads(result)
        
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        assert parsed["logger"] == "test_logger"
        assert "time" in parsed
        assert isinstance(parsed["time"], int)
    
    def test_formatting_with_extra(self):
        """Test JSON formatting with extra fields."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Error message",
            args=(),
            exc_info=None
        )
        record.extra = {"event": "test_event", "request_id": "123"}
        
        result = formatter.format(record)
        parsed = json.loads(result)
        
        assert parsed["level"] == "ERROR"
        assert parsed["message"] == "Error message"
        assert parsed["event"] == "test_event"
        assert parsed["request_id"] == "123"
    
    def test_unicode_handling(self):
        """Test that Unicode characters are handled properly."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Mensagem com acentos: ção, ã, ü",
            args=(),
            exc_info=None
        )
        
        result = formatter.format(record)
        parsed = json.loads(result)
        
        assert "ção" in parsed["message"]
        assert "ã" in parsed["message"]
        assert "ü" in parsed["message"]


class TestSetupLogger:
    """Tests for setup_logger function."""
    
    def test_logger_creation(self):
        """Test that setup_logger creates a proper logger."""
        logger = setup_logger("test_rag")
        
        assert logger.name == "test_rag"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0].formatter, JsonFormatter)
    
    def test_logger_reuse(self):
        """Test that calling setup_logger twice returns same logger."""
        logger1 = setup_logger("test_reuse")
        logger2 = setup_logger("test_reuse")
        
        assert logger1 is logger2
        assert len(logger1.handlers) == 1  # Should not duplicate handlers
    
    def test_default_name(self):
        """Test that default logger name is 'rag'."""
        logger = setup_logger()
        assert logger.name == "rag"
    
    def test_logger_output(self):
        """Test that logger produces valid JSON output."""
        # Capture log output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter())
        
        logger = logging.getLogger("test_output")
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        logger.info("Test log message")
        
        output = stream.getvalue().strip()
        parsed = json.loads(output)
        
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test log message"
        assert parsed["logger"] == "test_output"