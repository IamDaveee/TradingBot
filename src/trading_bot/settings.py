from pathlib import Path
from typing import Any

import yaml
from pydantic import (
    BaseModel,
    Field,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApplicationConfig(BaseModel):
    name: str
    environment: str


class ExchangeConfig(BaseModel):
    name: str
    mode: str
    demo_trading: bool
    rest_base_url: str
    public_websocket_url: str


class MarketDataConfig(BaseModel):
    symbols: list[str]
    channels: list[str]
    stale_after_seconds: float = Field(gt=0)
    health_check_interval_seconds: float = Field(gt=0)
    initial_reconnect_delay_seconds: float = Field(gt=0)
    maximum_reconnect_delay_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_reconnect_delays(
        self,
    ) -> "MarketDataConfig":
        if self.maximum_reconnect_delay_seconds < self.initial_reconnect_delay_seconds:
            raise ValueError(
                "Maximum reconnect delay must be greater than or equal to the initial delay"
            )

        return self


class TradingConfig(BaseModel):
    enabled: bool
    live_trading_enabled: bool


class LoggingConfig(BaseModel):
    level: str
    json_output: bool


class StorageConfig(BaseModel):
    enabled: bool
    directory: Path
    processed_directory: Path
    candle_directory: Path
    flush_every_records: int = Field(gt=0)
    queue_max_size: int = Field(gt=0)
    shutdown_timeout_seconds: float = Field(gt=0)


class FileConfig(BaseModel):
    application: ApplicationConfig
    exchange: ExchangeConfig
    market_data: MarketDataConfig
    trading: TradingConfig
    storage: StorageConfig
    logging: LoggingConfig


class EnvironmentSettings(BaseSettings):
    environment: str = Field(
        default="development",
        validation_alias="BOT_ENVIRONMENT",
    )

    config_file: Path = Field(
        default=Path("config/development.yaml"),
        validation_alias="BOT_CONFIG_FILE",
    )

    okx_api_key: str | None = Field(
        default=None,
        validation_alias="OKX_API_KEY",
    )

    okx_api_secret: str | None = Field(
        default=None,
        validation_alias="OKX_API_SECRET",
    )

    okx_api_passphrase: str | None = Field(
        default=None,
        validation_alias="OKX_API_PASSPHRASE",
    )

    live_trading_enabled: bool = Field(
        default=False,
        validation_alias="LIVE_TRADING_ENABLED",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        content = yaml.safe_load(file)

    if not isinstance(content, dict):
        raise ValueError(f"Configuration file must contain a YAML object: {path}")

    return content


def load_settings() -> tuple[EnvironmentSettings, FileConfig]:
    environment = EnvironmentSettings()
    yaml_content = load_yaml_file(environment.config_file)
    file_config = FileConfig.model_validate(yaml_content)

    return environment, file_config
