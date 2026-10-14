from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading_bot.market_data.historical_candle import (
    HistoricalCandle,
)


def create_candle() -> HistoricalCandle:
    start = datetime(
        2026,
        6,
        1,
        12,
        0,
        tzinfo=UTC,
    )

    return HistoricalCandle(
        instrument_id="BTC-USDT",
        interval="1m",
        interval_start=start,
        interval_end=(start + timedelta(minutes=1)),
        open_price=Decimal("100"),
        high_price=Decimal("110"),
        low_price=Decimal("90"),
        close_price=Decimal("105"),
        base_volume=Decimal("12.5"),
        currency_volume=Decimal("12.5"),
        quote_volume=Decimal("1300"),
        is_complete=True,
    )


def test_accepts_valid_historical_candle() -> None:
    candle = create_candle()

    assert candle.instrument_id == "BTC-USDT"
    assert candle.is_complete


def test_rejects_invalid_ohlc_range() -> None:
    candle = create_candle()

    with pytest.raises(ValueError):
        HistoricalCandle(
            instrument_id=candle.instrument_id,
            interval=candle.interval,
            interval_start=candle.interval_start,
            interval_end=candle.interval_end,
            open_price=Decimal("120"),
            high_price=Decimal("110"),
            low_price=Decimal("90"),
            close_price=Decimal("105"),
            base_volume=candle.base_volume,
            currency_volume=(candle.currency_volume),
            quote_volume=candle.quote_volume,
            is_complete=True,
        )


def test_rejects_negative_volume() -> None:
    candle = create_candle()

    with pytest.raises(ValueError):
        HistoricalCandle(
            instrument_id=candle.instrument_id,
            interval=candle.interval,
            interval_start=candle.interval_start,
            interval_end=candle.interval_end,
            open_price=candle.open_price,
            high_price=candle.high_price,
            low_price=candle.low_price,
            close_price=candle.close_price,
            base_volume=Decimal("-1"),
            currency_volume=(candle.currency_volume),
            quote_volume=candle.quote_volume,
            is_complete=True,
        )
