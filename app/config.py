"""Application settings loaded from environment / .env file."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API keys
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    google_maps_api_key: str = Field(default="", alias="GOOGLE_MAPS_API_KEY")
    weather_api_key: str = Field(default="", alias="WEATHER_API_KEY")
    serpapi_api_key: str = Field(default="", alias="SERPAPI_API_KEY")
    currency_api_key: str = Field(default="", alias="CURRENCY_API_KEY")

    # App
    app_env: Literal["dev", "prod"] = Field(default="dev", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Groq model
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")

    # How agents reach the tools: "local" calls the registry in-process, "mcp"
    # goes through the MCP server (MCP_SERVER_URL if set, else spawned over stdio).
    tool_transport: Literal["local", "mcp"] = Field(default="local", alias="TOOL_TRANSPORT")
    mcp_server_url: str = Field(default="", alias="MCP_SERVER_URL")

    # Simulated latency for mock tools/LLM so mock runs behave like network calls.
    mock_tool_latency_ms: int = Field(default=10, alias="MOCK_TOOL_LATENCY_MS")
    mock_llm_latency_ms: int = Field(default=0, alias="MOCK_LLM_LATENCY_MS")

    @property
    def use_mock_fallback(self) -> bool:
        """True when mocks should be used (explicit flag OR all keys missing)."""
        import os

        explicit = os.getenv("USE_MOCK_FALLBACK", "").lower() in {"1", "true", "yes"}
        no_keys = not (
            self.groq_api_key
            or self.google_maps_api_key
            or self.weather_api_key
            or self.serpapi_api_key
            or self.currency_api_key
        )
        return explicit or no_keys


settings = Settings()