"""AI client abstraction supporting multiple providers."""

import asyncio
import os
from abc import ABC, abstractmethod
from typing import Optional, Sequence

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI, AsyncAzureOpenAI
from google import genai
from google.genai import types

from ..models import AIConfig, AIProvider, Config
from .tokens import record_usage


class AIError(Exception):
    """Exception raised when an AI API call fails."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        model: str,
        name: Optional[str] = None,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model
        self.name = name
        self.status_code = status_code
        self.error_code = error_code
        display_name = name or model
        detail = f"[{provider}] {display_name}"
        if status_code:
            detail += f" (HTTP {status_code})"
        if error_code:
            detail += f" - {error_code}"
        detail += f": {message}"
        super().__init__(detail)


class AIClient(ABC):
    """Abstract base class for AI clients."""

    @abstractmethod
    async def complete(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate completion from AI model.

        Args:
            system: System prompt
            user: User prompt
            temperature: Optional sampling temperature override
            max_tokens: Optional maximum tokens override

        Returns:
            str: Generated completion text
        """
        pass


class AIClients(AIClient):
    """Round-robin pool of AI clients with rate-limit failover."""

    def __init__(self, clients: Sequence[AIClient]):
        if not clients:
            raise ValueError("AIClients requires at least one AIClient")
        self.clients = list(clients)
        self.config = self.clients[0].config
        self._next_index = 0
        self._lock = asyncio.Lock()

    async def complete(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        start_index = await self._reserve_client_index()
        last_rate_limit_error: Optional[Exception] = None

        for offset in range(len(self.clients)):
            client_index = (start_index + offset) % len(self.clients)
            client = self.clients[client_index]
            try:
                response = await client.complete(
                    system=system,
                    user=user,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                await self._mark_client_success(client_index)
                return response
            except Exception as exc:
                # Print detailed error info for debugging
                error_detail = str(exc)
                if isinstance(exc, AIError):
                    error_detail = exc.args[0] if exc.args else str(exc)
                client_name = getattr(client, 'config', None)
                model_name = getattr(client, 'model', 'unknown') if client_name else 'unknown'
                provider_name = getattr(client_name, 'provider', 'unknown') if client_name else 'unknown'
                alias_name = getattr(client_name, 'name', None) if client_name else None
                display_name = alias_name or model_name
                print(f"[{provider_name}] {display_name}: {error_detail}")
                last_rate_limit_error = exc

        if last_rate_limit_error is not None:
            raise last_rate_limit_error
        raise RuntimeError("No AI clients available")

    async def _reserve_client_index(self) -> int:
        async with self._lock:
            index = self._next_index
            self._next_index = (self._next_index + 1) % len(self.clients)
            return index

    async def _mark_client_success(self, index: int) -> None:
        async with self._lock:
            self._next_index = (index + 1) % len(self.clients)


class AnthropicClient(AIClient):
    """Client for Anthropic Claude models."""

    def __init__(self, config: AIConfig):
        """Initialize Anthropic client.

        Args:
            config: AI configuration
        """
        self.config = config

        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key: {config.api_key_env}")

        kwargs = {"api_key": api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url

        self.client = AsyncAnthropic(**kwargs)
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    async def complete(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate completion using Claude.

        Args:
            system: System prompt
            user: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            str: Generated text
        """
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens

        try:
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}]
            )
        except Exception as exc:
            print("error here")
            raise AIError(
                _extract_error_message(exc),
                provider="anthropic",
                model=self.model,
                name=self.config.name,
                status_code=_extract_status_code(exc),
                error_code=_extract_error_code(exc),
            ) from exc

        usage = getattr(message, "usage", None)
        if usage is not None:
            record_usage(
                "anthropic",
                input_tokens=getattr(usage, "input_tokens", 0),
                output_tokens=getattr(usage, "output_tokens", 0),
            )
        return message.content[0].text


class OpenAIClient(AIClient):
    """Client for OpenAI models."""

    def __init__(self, config: AIConfig):
        """Initialize OpenAI client.

        Args:
            config: AI configuration
        """
        self.config = config

        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key: {config.api_key_env}")

        kwargs = {"api_key": api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url

        self.client = AsyncOpenAI(**kwargs)
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    async def complete(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate completion using OpenAI.

        Args:
            system: System prompt
            user: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            str: Generated text
        """
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
        except Exception as exc:
            raise AIError(
                _extract_error_message(exc),
                provider="openai",
                model=self.model,
                name=self.config.name,
                status_code=_extract_status_code(exc),
                error_code=_extract_error_code(exc),
            ) from exc

        usage = getattr(response, "usage", None)
        if usage is not None:
            record_usage(
                "openai",
                input_tokens=getattr(usage, "prompt_tokens", 0),
                output_tokens=getattr(usage, "completion_tokens", 0),
            )
        return response.choices[0].message.content


class AzureOpenAIClient(AIClient):
    """Client for Azure OpenAI deployments.

    Uses the native AsyncAzureOpenAI client, which requires the deployment
    name (passed as `model`), azure_endpoint (resource base URL), and
    api_version. The deployment path is assembled internally by the SDK.
    """

    # Newer reasoning-series models reject legacy `max_tokens` and require
    # `max_completion_tokens` instead. Azure uses deployment names as `model`,
    # so a best-effort guess can be wrong for custom deployment aliases.
    _MODELS_REQUIRING_MAX_COMPLETION_TOKENS = ("o1", "o3", "o4", "gpt-5")

    def __init__(self, config: AIConfig):
        """Initialize Azure OpenAI client.

        Args:
            config: AI configuration
        """
        self.config = config

        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key: {config.api_key_env}")
        if not config.azure_endpoint_env:
            raise ValueError("azure_endpoint_env is required for azure provider")
        azure_endpoint = os.getenv(config.azure_endpoint_env)
        if not azure_endpoint:
            raise ValueError(f"Missing Azure endpoint: {config.azure_endpoint_env}")
        if not config.api_version:
            raise ValueError("api_version is required for azure provider")

        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=config.api_version,
        )
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self._use_max_completion_tokens = any(
            config.model.startswith(prefix)
            for prefix in self._MODELS_REQUIRING_MAX_COMPLETION_TOKENS
        )

    async def complete(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate completion using Azure OpenAI.

        Args:
            system: System prompt
            user: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            str: Generated text
        """
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens

        try:
            response = await self._create_completion(
                system=system,
                user=user,
                temperature=temperature,
                max_tokens=max_tokens,
                use_max_completion_tokens=self._use_max_completion_tokens,
            )
        except Exception as exc:
            fallback = self._token_fallback_mode(str(exc))
            if fallback is None:
                raise AIError(
                    _extract_error_message(exc),
                    provider="azure",
                    model=self.model,
                    name=self.config.name,
                    status_code=_extract_status_code(exc),
                    error_code=_extract_error_code(exc),
                ) from exc

            self._use_max_completion_tokens = fallback
            try:
                response = await self._create_completion(
                    system=system,
                    user=user,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    use_max_completion_tokens=fallback,
                )
            except Exception as exc2:
                raise AIError(
                    _extract_error_message(exc2),
                    provider="azure",
                    model=self.model,
                    name=self.config.name,
                    status_code=_extract_status_code(exc2),
                    error_code=_extract_error_code(exc2),
                ) from exc2

        usage = getattr(response, "usage", None)
        if usage is not None:
            record_usage(
                "openai",
                input_tokens=getattr(usage, "prompt_tokens", 0),
                output_tokens=getattr(usage, "completion_tokens", 0),
            )
        return response.choices[0].message.content

    async def _create_completion(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        use_max_completion_tokens: bool,
    ):
        tokens_kwarg = (
            {"max_completion_tokens": max_tokens}
            if use_max_completion_tokens
            else {"max_tokens": max_tokens}
        )
        return await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
            **tokens_kwarg,
        )

    @staticmethod
    def _token_fallback_mode(message: str) -> Optional[bool]:
        lowered = message.lower()
        if "max_completion_tokens" in lowered and "max_tokens" in lowered:
            return True
        if "max_tokens" in lowered and "max_completion_tokens" not in lowered:
            return False
        return None


class MiniMaxClient(AIClient):
    """Client for MiniMax models via OpenAI-compatible API."""

    def __init__(self, config: AIConfig):
        """Initialize MiniMax client.

        Args:
            config: AI configuration
        """
        self.config = config

        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key: {config.api_key_env}")

        kwargs = {
            "api_key": api_key,
            "base_url": config.base_url or "https://api.minimax.io/v1",
        }

        self.client = AsyncOpenAI(**kwargs)
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    async def complete(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate completion using MiniMax.

        MiniMax requires temperature in (0.0, 1.0] and does not support
        response_format, so we rely on prompt engineering for JSON output.

        Args:
            system: System prompt
            user: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            str: Generated text
        """
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens

        # MiniMax temperature must be in (0.0, 1.0]; clamp 0 to a small value
        if temperature <= 0:
            temperature = 0.01

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise AIError(
                _extract_error_message(exc),
                provider="minimax",
                model=self.model,
                name=self.config.name,
                status_code=_extract_status_code(exc),
                error_code=_extract_error_code(exc),
            ) from exc

        usage = getattr(response, "usage", None)
        if usage is not None:
            record_usage(
                "minimax",
                input_tokens=getattr(usage, "prompt_tokens", 0),
                output_tokens=getattr(usage, "completion_tokens", 0),
            )
        return response.choices[0].message.content


class AliClient(AIClient):
    """Client for Alibaba DashScope (OpenAI-compatible API)."""

    def __init__(self, config: AIConfig):
        """Initialize DashScope client.

        Args:
            config: AI configuration
        """
        self.config = config

        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key: {config.api_key_env}")

        kwargs = {
            "api_key": api_key,
            "base_url": config.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        }
        self.client = AsyncOpenAI(**kwargs)
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    async def complete(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate completion using DashScope.

        Args:
            system: System prompt
            user: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            str: Generated text
        """
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
        except Exception as exc:
            raise AIError(
                _extract_error_message(exc),
                provider="ali",
                model=self.model,
                name=self.config.name,
                status_code=_extract_status_code(exc),
                error_code=_extract_error_code(exc),
            ) from exc

        return response.choices[0].message.content


class GeminiClient(AIClient):
    """Client for Google Gemini models."""

    def __init__(self, config: AIConfig):
        """Initialize Gemini client.

        Args:
            config: AI configuration
        """
        self.config = config

        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key: {config.api_key_env}")

        self.client = genai.Client(api_key=api_key)
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    async def complete(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate completion using Gemini.

        Args:
            system: System prompt
            user: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            str: Generated text
        """
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json"
                )
            )
        except Exception as exc:
            raise AIError(
                _extract_error_message(exc),
                provider="gemini",
                model=self.model,
                name=self.config.name,
                status_code=_extract_status_code(exc),
                error_code=_extract_error_code(exc),
            ) from exc

        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            total = getattr(usage, "total_token_count", 0) or 0
            prompt = getattr(usage, "prompt_token_count", 0) or 0
            completion = max(0, total - prompt)
            record_usage("gemini", input_tokens=prompt, output_tokens=completion)
        return response.text


def _extract_error_message(exc: Exception) -> str:
    """Extract a human-readable error message from an exception."""
    # Try to get a clean message from the exception
    message = str(exc)

    # For OpenAI SDK exceptions
    if hasattr(exc, "message"):
        message = exc.message

    # For Anthropic exceptions
    if hasattr(exc, "body"):
        body = getattr(exc, "body", None)
        if body and hasattr(body, "get"):
            error_info = body.get("error", {})
            if isinstance(error_info, dict):
                error_msg = error_info.get("message", "")
                if error_msg:
                    message = error_msg

    # Strip generic prefixes
    for prefix in ["Error:", "Exception:", "APIError:", "BadRequestError:", "NotFoundError:", "ServerError:", "TypeError:"]:
        if message.startswith(prefix):
            message = message[len(prefix):].strip()

    return message or "Unknown error"


def _extract_status_code(exc: Exception) -> Optional[int]:
    """Extract HTTP status code from an exception."""
    # Direct status_code attribute
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return status_code

    # Check response object
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if status_code is not None:
            return status_code

    # For OpenAI SDK exceptions
    if hasattr(exc, "status_code"):
        return exc.status_code

    # Try to parse from message
    message = str(exc)
    import re
    match = re.search(r"\b(\d{3})\b", message)
    if match:
        code = int(match.group(1))
        if 100 <= code < 600:
            return code

    return None


def _extract_error_code(exc: Exception) -> Optional[str]:
    """Extract error code string from an exception."""
    # For OpenAI SDK exceptions
    code = getattr(exc, "code", None)
    if code:
        return str(code)

    # For Anthropic exceptions
    if hasattr(exc, "body"):
        body = getattr(exc, "body", None)
        if body and hasattr(body, "get"):
            error_info = body.get("error", {})
            if isinstance(error_info, dict):
                return error_info.get("type") or error_info.get("code")

    # Try to extract from message
    message = str(exc)
    import re
    match = re.search(r"\[([A-Z_]+)\]", message)
    if match:
        return match.group(1)

    return None


def     _is_rate_limit_error(exc: Exception) -> bool:
    """Return True when an SDK exception represents HTTP 429/rate limiting."""
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True

    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) == 429:
        return True

    code = str(getattr(exc, "code", "")).lower()
    message = str(exc).lower()
    return (
        ("rate" in message and ("limit" in message or "limited" in message))
        or "429" in message
        or "too many requests" in message
        or code in {"rate_limit", "rate_limited", "rate_limit_exceeded"}
    )


def create_ai_clients(config: Config) -> AIClients:
    """Create a round-robin AI client pool from app configuration.

    `ai_providers` takes precedence over the legacy single-provider `ai`
    field. When only `ai` is configured, the returned pool contains one client.
    """
    return AIClients([create_ai_client(ai_config) for ai_config in config.active_ai_configs])


def create_ai_client(config: AIConfig) -> AIClient:
    """Factory function to create appropriate AI client.

    Args:
        config: AI configuration

    Returns:
        AIClient: Initialized AI client

    Raises:
        ValueError: If provider is not supported
    """
    if config.provider == AIProvider.ANTHROPIC:
        return AnthropicClient(config)
    elif config.provider == AIProvider.OPENAI:
        return OpenAIClient(config)
    elif config.provider == AIProvider.AZURE:
        return AzureOpenAIClient(config)
    elif config.provider == AIProvider.ALI:
        return AliClient(config)
    elif config.provider == AIProvider.GEMINI:
        return GeminiClient(config)
    elif config.provider == AIProvider.DOUBAO:
        return OpenAIClient(config)
    elif config.provider == AIProvider.MINIMAX:
        return MiniMaxClient(config)
    else:
        raise ValueError(f"Unsupported AI provider: {config.provider}")
