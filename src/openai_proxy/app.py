"""FastAPI application for the OpenAI API Proxy."""

import logging
import os
import time
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import StreamingResponse

from .client import get_upstream_client
from .config import get_settings, Settings
from .mcp_client import initialize_mcp_client, shutdown_mcp_client
from .models import get_model_registry, reload_model_registry, ModelRegistry
from .proxy import process_chat_completion, process_chat_completion_stream
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelListResponse,
    ModelObject,
)
from .skills import get_skills_loader, reload_skills_loader


def setup_logging() -> logging.Logger:
    """Configure logging based on debug flag."""
    debug = os.environ.get("OPENAI_PROXY_DEBUG", "").lower() in ("1", "true", "yes")
    level = logging.DEBUG if debug else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (only in debug mode)
    if debug:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "openai-proxy.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return logging.getLogger(__name__)


logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup: Load model configurations and skills
    logger.info("Starting OpenAI API Proxy...")
    
    registry = get_model_registry()
    logger.info(f"Loaded {len(registry.list_models())} models: {[m.name for m in registry.list_models()]}")
    
    skills_loader = get_skills_loader()
    logger.info(f"Loaded {len(skills_loader.get_global_skills())} global skills")
    
    # Initialize MCP client (connects to configured servers)
    mcp_client = await initialize_mcp_client()
    logger.info(f"Connected to {len(mcp_client.get_connected_servers())} MCP servers")
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown: Close MCP connections and upstream client
    logger.info("Shutting down...")
    await shutdown_mcp_client()
    client = get_upstream_client()
    await client.close()
    logger.info("Shutdown complete")


app = FastAPI(
    title="OpenAI API Proxy",
    description="A proxy server that augments OpenAI-compatible APIs with additional tools",
    version="0.1.0",
    lifespan=lifespan,
)


def get_settings_dep() -> Settings:
    """Dependency for getting settings."""
    return get_settings()


def get_registry_dep() -> ModelRegistry:
    """Dependency for getting the model registry."""
    return get_model_registry()


@app.get("/v1/models", response_model=None)
async def list_models(
    registry: Annotated[ModelRegistry, Depends(get_registry_dep)],
) -> ModelListResponse:
    """List all available models.
    
    Returns both:
    - Upstream models from the backend (OpenRouter, OpenAI, etc.)
    - Augmented models defined in conf/models/ (prefixed with augmented/)
    """
    all_models: list[ModelObject] = []
    
    # Fetch upstream models
    try:
        client = get_upstream_client()
        upstream_response = await client.list_models()
        
        # Parse upstream models
        for model_data in upstream_response.get("data", []):
            all_models.append(ModelObject(
                id=model_data.get("id", ""),
                created=model_data.get("created", int(time.time())),
                owned_by=model_data.get("owned_by", "upstream"),
            ))
    except Exception as e:
        logger.warning(f"Failed to fetch upstream models: {e}")
    
    # Add augmented models
    for model in registry.list_models():
        all_models.append(ModelObject(
            id=model.name,
            created=model.created or int(time.time()),
            owned_by=model.owned_by,
        ))
    
    return ModelListResponse(data=all_models)


@app.get("/v1/models/{model_id}")
async def get_model(
    model_id: str,
    registry: Annotated[ModelRegistry, Depends(get_registry_dep)],
) -> ModelObject:
    """Get a specific model."""
    model = registry.get(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    return ModelObject(
        id=model.name,
        created=model.created or int(time.time()),
        owned_by=model.owned_by,
    )


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
    registry: Annotated[ModelRegistry, Depends(get_registry_dep)],
    authorization: Annotated[str | None, Header()] = None,
) -> ChatCompletionResponse | StreamingResponse:
    """Create a chat completion.
    
    Supports two modes:
    1. Augmented models (augmented/*): Use model configs from conf/models/ with
       tools, MCP servers, skills, and system prompts
    2. Pass-through: Any other model name is forwarded directly to upstream
    """
    logger.info(f"Received chat completion request for model: {request.model}")
    logger.debug(f"Request details: stream={request.stream}, messages_count={len(request.messages)}")
    logger.debug(f"Client tools: {[t.function.name for t in request.tools] if request.tools else 'None'}")
    
    # Check if this is an augmented model
    model_config = registry.get(request.model)
    
    if model_config is None:
        # Pass-through mode: Forward directly to upstream without augmentation
        logger.info(f"Pass-through mode for model: {request.model}")
        return await _handle_passthrough(request)
    
    # Augmented mode: Apply tools, MCP, skills, etc.
    logger.info(f"Augmented mode: {model_config.name} -> upstream: {model_config.effective_upstream_model}")
    logger.debug(f"Model tools: {[t.function.name for t in model_config.tools]}")
    logger.debug(f"Model MCP servers: {model_config.mcp_servers}")
    logger.debug(f"Model skills: {model_config.skills}")

    try:
        if request.stream:
            logger.info("Processing as streaming request")
            return StreamingResponse(
                process_chat_completion_stream(request, model_config),
                media_type="text/event-stream",
            )
        else:
            logger.info("Processing as non-streaming request")
            response = await process_chat_completion(request, model_config)
            logger.info(f"Response received: finish_reason={response.choices[0].finish_reason if response.choices else 'N/A'}")
            logger.debug(f"Response content preview: {(response.choices[0].message.content or '')[:100] if response.choices else 'N/A'}...")
            return response
    except Exception as e:
        logger.exception(f"Error processing chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _handle_passthrough(request: ChatCompletionRequest) -> ChatCompletionResponse | StreamingResponse:
    """Handle pass-through requests to upstream without augmentation."""
    import json
    from .client import get_upstream_client
    
    client = get_upstream_client()
    
    try:
        if request.stream:
            async def stream_passthrough():
                async for chunk in client.chat_completion_stream(request):
                    if chunk == "[DONE]":
                        yield "data: [DONE]\n\n"
                    else:
                        yield f"data: {chunk}\n\n"
            
            return StreamingResponse(
                stream_passthrough(),
                media_type="text/event-stream",
            )
        else:
            return await client.chat_completion(request)
    except httpx.HTTPStatusError as e:
        logger.error(f"Upstream error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.exception(f"Error in pass-through: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/reload")
async def reload_config() -> dict:
    """Reload model configurations, skills, and reconnect MCP servers."""
    registry = reload_model_registry()
    skills_loader = reload_skills_loader()
    
    # Reconnect MCP servers
    await shutdown_mcp_client()
    mcp_client = await initialize_mcp_client()
    
    return {
        "status": "ok",
        "models_loaded": len(registry.list_models()),
        "skills_loaded": len(skills_loader.get_global_skills()),
        "mcp_servers_connected": len(mcp_client.get_connected_servers()),
    }


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "healthy"}
