"""Centralized logging configuration for the ReelMotion MCP server."""
import logging
import os


def setup_logging() -> None:
    """Configure logging for the application based on LOG_LEVEL env var.

    Set LOG_LEVEL=DEBUG in .env during development to see all messages.
    Production default is WARNING.
    """
    log_level_name = os.getenv("LOG_LEVEL", "WARNING").upper()
    log_level = getattr(logging, log_level_name, logging.WARNING)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Suppress verbose third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("google.generativeai").setLevel(logging.WARNING)
    logging.getLogger("google.auth").setLevel(logging.WARNING)
