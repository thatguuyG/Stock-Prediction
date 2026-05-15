"""Application settings loaded from environment / .env."""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = Field(
        default="postgresql+psycopg://stockpred:stockpred@localhost:5432/stockpred"
    )
    newsapi_key: str = Field(default="")
    watchlist_raw: str = Field(default="AAPL,MSFT,NVDA", alias="WATCHLIST")
    log_level: str = Field(default="INFO")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def watchlist(self) -> list[str]:
        return [s.strip().upper() for s in self.watchlist_raw.split(",") if s.strip()]


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings  # pylint: disable=global-statement
    if _settings is None:
        _settings = Settings()
    return _settings
