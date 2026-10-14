from datetime import UTC, datetime, timedelta
from decimal import Decimal

from trading_bot.indicators.atr import (
    AverageTrueRange,
)
from trading_bot.market_data.candle import (
    TickerCandle,
)


def candle(
    minute: int,
    *,
    high: str,
    low: str,
    close: str,
) -> TickerCandle:
    start = datetime(
        2026,
        5,
        1,
        tzinfo=UTC,
    ) + timedelta(minutes=minute)

    high_value = Decimal(high)
    low_value = Decimal(low)
    close_value = Decimal(close)

    return TickerCandle(
        instrument_id="BTC-USDT",
        interval_start=start,
        interval_end=(start + timedelta(minutes=1)),
        open_price=close_value,
        high_price=high_value,
        low_price=low_value,
        close_price=close_value,
        first_bid_price=close_value,
        last_bid_price=close_value,
        first_ask_price=close_value,
        last_ask_price=close_value,
        minimum_spread=Decimal("0"),
        maximum_spread=Decimal("0"),
        average_spread=Decimal("0"),
        observed_size=Decimal("1"),
        tick_count=1,
        first_exchange_timestamp=start,
        last_exchange_timestamp=start,
        first_received_timestamp=start,
        last_received_timestamp=start,
    )


def test_atr_is_not_ready_before_period() -> None:
    indicator = AverageTrueRange(period=3)

    assert (
        indicator.update(
            candle(
                0,
                high="12",
                low="10",
                close="11",
            )
        )
        is None
    )

    assert (
        indicator.update(
            candle(
                1,
                high="13",
                low="11",
                close="12",
            )
        )
        is None
    )


def test_calculates_simple_average_true_range() -> None:
    indicator = AverageTrueRange(period=3)

    indicator.update(
        candle(
            0,
            high="12",
            low="10",
            close="11",
        )
    )

    indicator.update(
        candle(
            1,
            high="14",
            low="12",
            close="13",
        )
    )

    result = indicator.update(
        candle(
            2,
            high="15",
            low="11",
            close="14",
        )
    )

    # True ranges:
    # Candle 1: 2
    # Candle 2: max(2, 3, 1) = 3
    # Candle 3: max(4, 2, 2) = 4
    assert result == Decimal("3")


def test_uses_rolling_period() -> None:
    indicator = AverageTrueRange(period=2)

    indicator.update(
        candle(
            0,
            high="11",
            low="10",
            close="10",
        )
    )

    assert indicator.update(
        candle(
            1,
            high="13",
            low="10",
            close="12",
        )
    ) == Decimal("2")

    assert indicator.update(
        candle(
            2,
            high="18",
            low="12",
            close="17",
        )
    ) == Decimal("4.5")
