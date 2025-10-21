try:
    from . import _message_transform
    from ._openai_client import (
        AZURE_OPENAI_USER_AGENT,
        AzureOpenAIChatCompletionClient,
        BaseOpenAIChatCompletionClient,
        OpenAIChatCompletionClient,
    )
    from .config import (
        AzureOpenAIClientConfigurationConfigModel,
        BaseOpenAIClientConfigurationConfigModel,
        CreateArgumentsConfigModel,
        OpenAIClientConfigurationConfigModel,
    )
except ImportError as e:
    raise ImportError(
        "Dependencies for OpenAI client not found. "
        "Please install the OpenAI package: "
        "pip install autogen-ext[openai]\n"
        f"Original error: {e}"
    ) from e

__all__ = [
    "OpenAIChatCompletionClient",
    "AzureOpenAIChatCompletionClient",
    "BaseOpenAIChatCompletionClient",
    "AzureOpenAIClientConfigurationConfigModel",
    "OpenAIClientConfigurationConfigModel",
    "BaseOpenAIClientConfigurationConfigModel",
    "CreateArgumentsConfigModel",
    "AZURE_OPENAI_USER_AGENT",
    "_message_transform",
]
