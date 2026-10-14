from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading_bot.backtesting.baseline import (
    calculate_buy_and_hold,
)
from trading_bot.market_data.candle import (
    TickerCandle,
)


def candle(
    minute: int,
    open_price: str,
    close_price: str,
    symbol: str = "BTC-USDT",
) -> TickerCandle:
    start = datetime(
        2026,
        6,
        27,
        12,
        minute,
        tzinfo=UTC,
    )

    open_value = Decimal(open_price)
    close_value = Decimal(close_price)

    return TickerCandle(
        instrument_id=symbol,
        interval_start=start,
        interval_end=(start + timedelta(minutes=1)),
        open_price=open_value,
        high_price=max(
            open_value,
            close_value,
        ),
        low_price=min(
            open_value,
            close_value,
        ),
        close_price=close_value,
        first_bid_price=open_value,
        last_bid_price=close_value,
        first_ask_price=open_value,
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


def test_buy_and_hold_without_costs() -> None:
    result = calculate_buy_and_hold(
        candles=[
            candle(0, "100", "100"),
            candle(1, "100", "120"),
        ],
        initial_cash=Decimal("1000"),
    )

    assert result.quantity == Decimal("10")
    assert result.final_equity == Decimal("1200")
    assert result.absolute_return == Decimal("200")
    assert result.return_percentage == Decimal("20.0")


def test_buy_and_hold_includes_fees() -> None:
    result = calculate_buy_and_hold(
        candles=[
            candle(0, "100", "100"),
            candle(1, "100", "100"),
        ],
        initial_cash=Decimal("1000"),
        fee_rate=Decimal("0.001"),
    )

    assert result.final_equity < Decimal("1000")
    assert result.entry_fee > 0
    assert result.exit_fee > 0


def test_buy_and_hold_rejects_multiple_symbols() -> None:
    with pytest.raises(ValueError):
        calculate_buy_and_hold(
            candles=[
                candle(
                    0,
                    "100",
                    "100",
                    symbol="BTC-USDT",
                ),
                candle(
                    1,
                    "100",
                    "100",
                    symbol="ETH-USDT",
                ),
            ],
            initial_cash=Decimal("1000"),
        )
