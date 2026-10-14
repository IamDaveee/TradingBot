from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading_bot.backtesting.coverage import (
    CandleCoverageAnalyzer,
    validate_minimum_history,
)
from trading_bot.market_data.candle import (
    TickerCandle,
)


def candle(
    minute: int,
) -> TickerCandle:
    start = datetime(
        2026,
        6,
        27,
        12,
        0,
        tzinfo=UTC,
    ) + timedelta(minutes=minute)

    price = Decimal("100")

    return TickerCandle(
        instrument_id="BTC-USDT",
        interval_start=start,
        interval_end=(start + timedelta(minutes=1)),
        open_price=price,
        high_price=price,
        low_price=price,
        close_price=price,
        first_bid_price=Decimal("99.9"),
        last_bid_price=Decimal("99.9"),
        first_ask_price=Decimal("100.1"),
        last_ask_price=Decimal("100.1"),
        minimum_spread=Decimal("0.2"),
        maximum_spread=Decimal("0.2"),
        average_spread=Decimal("0.2"),
        observed_size=Decimal("1"),
        tick_count=1,
        first_exchange_timestamp=start,
        last_exchange_timestamp=start,
        first_received_timestamp=start,
        last_received_timestamp=start,
    )


def test_detects_short_missing_gap() -> None:
    candles = [
        candle(0),
        candle(1),
        candle(4),
    ]

    report = CandleCoverageAnalyzer(session_gap_threshold_minutes=5).analyze(candles)

    assert report.total_missing_intervals == 2
    assert report.total_session_gaps == 0


def test_detects_large_session_gap() -> None:
    candles = [
        candle(0),
        candle(1),
        candle(20),
    ]

    report = CandleCoverageAnalyzer(session_gap_threshold_minutes=5).analyze(candles)

    assert report.total_missing_intervals == 0
    assert report.total_session_gaps == 1


def test_detects_duplicate_intervals() -> None:
    candles = [
        candle(0),
        candle(0),
    ]

    report = CandleCoverageAnalyzer().analyze(candles)

    assert report.total_duplicates == 1


def test_minimum_history_passes() -> None:
    validate_minimum_history(
        candles=[
            candle(0),
            candle(1),
            candle(2),
        ],
        required_candles=3,
    )


def test_minimum_history_rejects_short_data() -> None:
    with pytest.raises(
        ValueError,
        match="Insufficient candle history",
    ):
        validate_minimum_history(
            candles=[
                candle(0),
                candle(1),
            ],
            required_candles=3,
        )
