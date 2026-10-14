from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import cast

import pytest

from trading_bot.exchanges.okx.models import OkxTicker
from trading_bot.exchanges.okx.public_websocket import (
    OkxPublicWebSocketClient,
)
from trading_bot.market_data.service import (
    MarketDataService,
    MarketDataUnavailableError,
)


def create_ticker(
    symbol: str,
    received_timestamp: datetime,
) -> OkxTicker:
    return OkxTicker(
        instrument_id=symbol,
        last_price=Decimal("100"),
        last_size=Decimal("1"),
        best_ask_price=Decimal("100.1"),
        best_ask_size=Decimal("2"),
        best_bid_price=Decimal("99.9"),
        best_bid_size=Decimal("3"),
        exchange_timestamp=received_timestamp,
        received_timestamp=received_timestamp,
    )


def create_service() -> MarketDataService:
    websocket_client = cast(
        OkxPublicWebSocketClient,
        object(),
    )

    return MarketDataService(
        websocket_client=websocket_client,
        symbols=[
            "BTC-USDT",
            "ETH-USDT",
            "SOL-USDT",
        ],
        stale_after_seconds=10,
    )


@pytest.mark.asyncio
async def test_service_stores_latest_ticker() -> None:
    service = create_service()

    timestamp = datetime(
        2026,
        6,
        27,
        12,
        0,
        tzinfo=UTC,
    )

    ticker = create_ticker(
        "BTC-USDT",
        timestamp,
    )

    await service.update_ticker(ticker)

    stored_ticker = await service.get_latest_ticker("BTC-USDT")

    assert stored_ticker == ticker
    assert await service.has_data("BTC-USDT") is True


@pytest.mark.asyncio
async def test_service_replaces_older_ticker() -> None:
    service = create_service()

    first_ticker = create_ticker(
        "BTC-USDT",
        datetime(
            2026,
            6,
            27,
            12,
            0,
            tzinfo=UTC,
        ),
    )

    second_ticker = create_ticker(
        "BTC-USDT",
        datetime(
            2026,
            6,
            27,
            12,
            0,
            1,
            tzinfo=UTC,
        ),
    )

    await service.update_ticker(first_ticker)
    await service.update_ticker(second_ticker)

    stored_ticker = await service.get_latest_ticker("BTC-USDT")

    assert stored_ticker == second_ticker


@pytest.mark.asyncio
async def test_service_rejects_unknown_symbol() -> None:
    service = create_service()

    ticker = create_ticker(
        "DOGE-USDT",
        datetime.now(UTC),
    )

    await service.update_ticker(ticker)

    assert await service.has_data("DOGE-USDT") is False


@pytest.mark.asyncio
async def test_missing_ticker_raises_error() -> None:
    service = create_service()

    with pytest.raises(MarketDataUnavailableError):
        await service.get_latest_ticker("BTC-USDT")


@pytest.mark.asyncio
async def test_service_is_ready_after_all_symbols_arrive() -> None:
    service = create_service()

    timestamp = datetime.now(UTC)

    assert await service.is_ready() is False

    for symbol in service.symbols:
        await service.update_ticker(
            create_ticker(
                symbol,
                timestamp,
            )
        )

    assert await service.is_ready() is True
    assert await service.get_missing_symbols() == []


@pytest.mark.asyncio
async def test_service_reports_missing_symbols() -> None:
    service = create_service()

    await service.update_ticker(
        create_ticker(
            "BTC-USDT",
            datetime.now(UTC),
        )
    )

    missing_symbols = await service.get_missing_symbols()

    assert missing_symbols == [
        "ETH-USDT",
        "SOL-USDT",
    ]


@pytest.mark.asyncio
async def test_service_detects_stale_ticker() -> None:
    service = create_service()

    received_timestamp = datetime(
        2026,
        6,
        27,
        12,
        0,
        tzinfo=UTC,
    )

    await service.update_ticker(
        create_ticker(
            "BTC-USDT",
            received_timestamp,
        )
    )

    current_time = received_timestamp + timedelta(seconds=11)

    assert (
        await service.is_stale(
            "BTC-USDT",
            current_time=current_time,
        )
        is True
    )


@pytest.mark.asyncio
async def test_service_accepts_fresh_ticker() -> None:
    service = create_service()

    received_timestamp = datetime(
        2026,
        6,
        27,
        12,
        0,
        tzinfo=UTC,
    )

    await service.update_ticker(
        create_ticker(
            "BTC-USDT",
            received_timestamp,
        )
    )

    current_time = received_timestamp + timedelta(seconds=5)

    assert (
        await service.is_stale(
            "BTC-USDT",
            current_time=current_time,
        )
        is False
    )


@pytest.mark.asyncio
async def test_missing_symbols_are_also_stale() -> None:
    service = create_service()

    timestamp = datetime(
        2026,
        6,
        27,
        12,
        0,
        tzinfo=UTC,
    )

    await service.update_ticker(
        create_ticker(
            "BTC-USDT",
            timestamp,
        )
    )

    stale_symbols = await service.get_stale_symbols(
        current_time=timestamp,
    )

    assert stale_symbols == [
        "ETH-USDT",
        "SOL-USDT",
    ]


def test_service_rejects_empty_symbol_list() -> None:
    websocket_client = cast(
        OkxPublicWebSocketClient,
        object(),
    )

    with pytest.raises(ValueError):
        MarketDataService(
            websocket_client=websocket_client,
            symbols=[],
            stale_after_seconds=10,
        )


def test_service_rejects_invalid_stale_threshold() -> None:
    websocket_client = cast(
        OkxPublicWebSocketClient,
        object(),
    )

    with pytest.raises(ValueError):
        MarketDataService(
            websocket_client=websocket_client,
            symbols=["BTC-USDT"],
            stale_after_seconds=0,
        )
