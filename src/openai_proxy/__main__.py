"""Main entry point for the OpenAI API Proxy."""

import argparse
import os

import uvicorn

from .config import get_settings


def main():
    """Run the proxy server."""
    parser = argparse.ArgumentParser(description="OpenAI API Proxy")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (verbose output and file logging)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config directory path (default: ~/.config/openai_proxy)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind the server (overrides config)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind the server (overrides config)",
    )
    args = parser.parse_args()
    
    # Set environment variables before loading settings
    if args.debug:
        os.environ["OPENAI_PROXY_DEBUG"] = "1"
    if args.config:
        os.environ["OPENAI_PROXY_CONFIG_DIR"] = args.config
    
    settings = get_settings()
    host = args.host or settings.host
    port = args.port or settings.port
    
    uvicorn.run(
        "openai_proxy.app:app",
        host=host,
        port=port,
        reload=args.debug,
        log_level="debug" if args.debug else "info",
    )


if __name__ == "__main__":
    main()
