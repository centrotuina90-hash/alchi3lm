try:
    from ._azure_ai_client import AzureAIChatCompletionClient
    from .config import AzureAIChatCompletionClientConfig
except ImportError as e:
    raise ImportError(
        "Dependencies for Azure AI client not found. "
        "Please install the Azure AI Inference package: "
        "pip install autogen-ext[azure]\n"
        f"Original error: {e}"
    ) from e

__all__ = ["AzureAIChatCompletionClient", "AzureAIChatCompletionClientConfig"]
