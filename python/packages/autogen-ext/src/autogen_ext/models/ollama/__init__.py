try:
    from ._ollama_client import OllamaChatCompletionClient
    from .config import (
        BaseOllamaClientConfigurationConfigModel,
        CreateArgumentsConfigModel,
    )
except ImportError as e:
    raise ImportError(
        "Dependencies for Ollama client not found. "
        "Please install the Ollama package: "
        "pip install autogen-ext[ollama]\n"
        f"Original error: {e}"
    ) from e

__all__ = [
    "OllamaChatCompletionClient",
    "BaseOllamaClientConfigurationConfigModel",
    "CreateArgumentsConfigModel",
]
