from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading_bot.backtesting.models import (
    OrderSide,
)
from trading_bot.backtesting.portfolio import (
    Position,
)
from trading_bot.backtesting.strategy import (
    StrategyContext,
)
from trading_bot.market_data.candle import (
    TickerCandle,
)
from trading_bot.strategies.moving_average_crossover import (
    MovingAverageCrossoverStrategy,
)


def create_candle(
    minute: int,
    close_price: str,
) -> TickerCandle:
    start = datetime(
        2026,
        6,
        27,
        12,
        minute,
        tzinfo=UTC,
    )

    price = Decimal(close_price)

    return TickerCandle(
        instrument_id="BTC-USDT",
        interval_start=start,
        interval_end=(start + timedelta(minutes=1)),
        open_price=price,
        high_price=price,
        low_price=price,
        close_price=price,
        first_bid_price=(price - Decimal("0.1")),
        last_bid_price=(price - Decimal("0.1")),
        first_ask_price=(price + Decimal("0.1")),
        last_ask_price=(price + Decimal("0.1")),
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


def create_context(
    quantity: str = "0",
    cash: str = "1000",
) -> StrategyContext:
    return StrategyContext(
        cash=Decimal(cash),
        equity=Decimal(cash),
        position=Position(
            instrument_id="BTC-USDT",
            quantity=Decimal(quantity),
            average_entry_price=(Decimal("100") if Decimal(quantity) > 0 else Decimal("0")),
        ),
    )


def test_strategy_buys_on_cross_above() -> None:
    strategy = MovingAverageCrossoverStrategy(
        fast_window=2,
        slow_window=3,
        allocation_fraction=Decimal("0.5"),
    )

    candles = [
        create_candle(0, "100"),
        create_candle(1, "99"),
        create_candle(2, "98"),
        create_candle(3, "110"),
    ]

    orders = []

    for candle in candles:
        orders.extend(
            strategy.on_candle(
                candle,
                create_context(),
            )
        )

    assert len(orders) == 1
    assert orders[0].side == OrderSide.BUY
    assert orders[0].created_at == (candles[-1].interval_end)


def test_strategy_sells_on_cross_below() -> None:
    strategy = MovingAverageCrossoverStrategy(
        fast_window=2,
        slow_window=3,
        allocation_fraction=Decimal("0.5"),
    )

    prices = [
        "100",
        "101",
        "110",
        "90",
    ]

    orders = []

    for index, price in enumerate(prices):
        context = create_context(quantity=("1" if index == len(prices) - 1 else "0"))

        orders.extend(
            strategy.on_candle(
                create_candle(
                    index,
                    price,
                ),
                context,
            )
        )

    assert len(orders) == 1
    assert orders[0].side == OrderSide.SELL
    assert orders[0].quantity == Decimal("1")


def test_strategy_rejects_invalid_windows() -> None:
    with pytest.raises(ValueError):
        MovingAverageCrossoverStrategy(
            fast_window=3,
            slow_window=3,
            allocation_fraction=Decimal("0.5"),
        )


def test_strategy_rejects_invalid_allocation() -> None:
    with pytest.raises(ValueError):
        MovingAverageCrossoverStrategy(
            fast_window=2,
            slow_window=3,
            allocation_fraction=Decimal("1.1"),
        )
