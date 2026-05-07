import asyncio
from types import SimpleNamespace

from src.ai.client import AIClients
from src.models import Config


class FakeRateLimitError(Exception):
    status_code = 429


class FakeClient:
    def __init__(self, name: str, responses: list[str | Exception] | None = None):
        self.name = name
        self.config = SimpleNamespace(throttle_sec=0.0)
        self.responses = responses or [name]
        self.calls = 0

    async def complete(self, **kwargs):
        self.calls += 1
        response = self.responses.pop(0) if self.responses else self.name
        if isinstance(response, Exception):
            raise response
        return response


def test_ai_clients_round_robin_between_clients():
    pool = AIClients([FakeClient("a"), FakeClient("b")])

    assert asyncio.run(pool.complete(system="s", user="u")) == "a"
    assert asyncio.run(pool.complete(system="s", user="u")) == "b"
    assert asyncio.run(pool.complete(system="s", user="u")) == "a"


def test_ai_clients_falls_back_to_next_client_on_429():
    first = FakeClient("a", [FakeRateLimitError("rate limited")])
    second = FakeClient("b", ["ok"])
    pool = AIClients([first, second])

    assert asyncio.run(pool.complete(system="s", user="u")) == "ok"
    assert first.calls == 1
    assert second.calls == 1


def test_config_prefers_ai_providers_over_legacy_ai():
    config = Config.model_validate(
        {
            "ai": {
                "provider": "openai",
                "model": "legacy",
                "api_key_env": "LEGACY_API_KEY",
            },
            "ai_providers": [
                {
                    "provider": "openai",
                    "model": "primary",
                    "api_key_env": "PRIMARY_API_KEY",
                    "languages": ["zh"],
                },
                {
                    "provider": "anthropic",
                    "model": "fallback",
                    "api_key_env": "FALLBACK_API_KEY",
                },
            ],
            "sources": {},
            "filtering": {},
        }
    )

    assert [item.model for item in config.active_ai_configs] == ["primary", "fallback"]
    assert config.primary_ai.model == "primary"
    assert config.primary_ai.languages == ["zh"]
