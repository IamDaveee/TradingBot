from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading_bot.backtesting.sweep import (
    MovingAverageParameterSweep,
    SweepConfiguration,
    create_window_configurations,
)
from trading_bot.market_data.candle import (
    TickerCandle,
)


def candle(
    minute: int,
    price: str,
) -> TickerCandle:
    start = datetime(
        2026,
        6,
        27,
        12,
        0,
        tzinfo=UTC,
    ) + timedelta(minutes=minute)

    value = Decimal(price)

    return TickerCandle(
        instrument_id="BTC-USDT",
        interval_start=start,
        interval_end=(start + timedelta(minutes=1)),
        open_price=value,
        high_price=value,
        low_price=value,
        close_price=value,
        first_bid_price=(value - Decimal("0.1")),
        last_bid_price=(value - Decimal("0.1")),
        first_ask_price=(value + Decimal("0.1")),
        last_ask_price=(value + Decimal("0.1")),
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


def test_creates_valid_window_combinations() -> None:
    configurations = create_window_configurations(
        fast_windows=[2, 5],
        slow_windows=[3, 10],
    )

    assert configurations == [
        SweepConfiguration(
            fast_window=2,
            slow_window=3,
        ),
        SweepConfiguration(
            fast_window=2,
            slow_window=10,
        ),
        SweepConfiguration(
            fast_window=5,
            slow_window=10,
        ),
    ]


def test_sweep_returns_one_result_per_configuration() -> None:
    prices = [
        "100",
        "99",
        "98",
        "110",
        "111",
        "90",
        "89",
        "100",
        "105",
    ]

    candles = [candle(index, price) for index, price in enumerate(prices)]

    configurations = [
        SweepConfiguration(
            fast_window=1,
            slow_window=2,
        ),
        SweepConfiguration(
            fast_window=2,
            slow_window=3,
        ),
    ]

    sweep = MovingAverageParameterSweep(
        initial_cash=Decimal("1000"),
        allocation_fraction=Decimal("0.5"),
        fee_rate=Decimal("0"),
        slippage_basis_points=Decimal("0"),
    )

    results = sweep.run(
        candles=candles,
        configurations=configurations,
    )

    assert len(results) == 2

    result_pairs = {
        (
            result.fast_window,
            result.slow_window,
        )
        for result in results
    }

    assert result_pairs == {
        (1, 2),
        (2, 3),
    }


def test_sweep_rejects_invalid_configuration() -> None:
    sweep = MovingAverageParameterSweep(
        initial_cash=Decimal("1000"),
        allocation_fraction=Decimal("0.5"),
        fee_rate=Decimal("0"),
        slippage_basis_points=Decimal("0"),
    )

    with pytest.raises(ValueError):
        sweep.run(
            candles=[
                candle(0, "100"),
                candle(1, "101"),
            ],
            configurations=[
                SweepConfiguration(
                    fast_window=3,
                    slow_window=2,
                )
            ],
        )
