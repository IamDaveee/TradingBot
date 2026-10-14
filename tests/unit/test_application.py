from pathlib import Path

from trading_bot.settings import load_settings


def test_settings_can_be_loaded() -> None:
    environment, config = load_settings()

    assert environment.environment == "development"
    assert config.application.name == "crypto-trading-bot"
    assert config.exchange.name == "okx"
    assert config.exchange.mode == "public"
    assert config.exchange.demo_trading is True
    assert config.exchange.rest_base_url == "https://www.okx.com"

    assert config.exchange.public_websocket_url == "wss://ws.okx.com:8443/ws/v5/public"
    assert config.market_data.channels == ["tickers"]

    assert config.market_data.symbols == [
        "BTC-USDT",
        "ETH-USDT",
        "SOL-USDT",
    ]

    assert config.trading.enabled is False
    assert environment.live_trading_enabled is False

    assert config.market_data.initial_reconnect_delay_seconds == 1
    assert config.market_data.maximum_reconnect_delay_seconds == 30

    assert config.market_data.stale_after_seconds == 10
    assert config.market_data.health_check_interval_seconds == 5

    assert config.storage.enabled is True
    assert config.storage.directory == Path("data/raw/tickers")
    assert config.storage.flush_every_records == 10

    assert config.storage.queue_max_size == 10000
    assert config.storage.shutdown_timeout_seconds == 10

    assert config.storage.processed_directory == Path("data/processed/tickers")

    assert config.storage.candle_directory == Path("data/processed/candles")
