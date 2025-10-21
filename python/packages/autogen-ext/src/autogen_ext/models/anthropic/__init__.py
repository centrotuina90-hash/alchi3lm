try:
    from ._anthropic_client import (
        AnthropicBedrockChatCompletionClient,
        AnthropicChatCompletionClient,
        BaseAnthropicChatCompletionClient,
    )
    from .config import (
        AnthropicBedrockClientConfiguration,
        AnthropicBedrockClientConfigurationConfigModel,
        AnthropicClientConfiguration,
        AnthropicClientConfigurationConfigModel,
        BedrockInfo,
        CreateArgumentsConfigModel,
    )
except ImportError as e:
    raise ImportError(
        "Dependencies for Anthropic client not found. "
        "Please install the Anthropic package: "
        "pip install autogen-ext[anthropic]\n"
        f"Original error: {e}"
    ) from e

__all__ = [
    "AnthropicChatCompletionClient",
    "AnthropicBedrockChatCompletionClient",
    "BaseAnthropicChatCompletionClient",
    "AnthropicClientConfiguration",
    "AnthropicBedrockClientConfiguration",
    "AnthropicClientConfigurationConfigModel",
    "AnthropicBedrockClientConfigurationConfigModel",
    "CreateArgumentsConfigModel",
    "BedrockInfo",
]
