"""LLM client abstraction for OpenAI-compatible chat completions."""

from aieng.syn_data.synbench.llm.client import LLMClient, LLMResponse, get_client
from aieng.syn_data.synbench.llm.config import DEFAULT_MODEL, get_model


__all__ = [
    "DEFAULT_MODEL",
    "LLMClient",
    "LLMResponse",
    "get_client",
    "get_model",
]
