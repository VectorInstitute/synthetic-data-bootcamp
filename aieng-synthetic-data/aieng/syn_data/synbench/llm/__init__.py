"""LLM client abstraction with real and mock backends."""

from aieng.syn_data.synbench.llm.client import LLMClient, LLMResponse, get_client
from aieng.syn_data.synbench.llm.config import DEFAULT_MODEL, get_model, is_mock_llm


__all__ = [
    "DEFAULT_MODEL",
    "LLMClient",
    "LLMResponse",
    "get_client",
    "get_model",
    "is_mock_llm",
]
