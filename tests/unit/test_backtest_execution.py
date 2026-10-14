from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading_bot.backtesting.execution import (
    BacktestExecutionEngine,
)
from trading_bot.backtesting.fees import (
    PercentageFeeModel,
)
from trading_bot.backtesting.models import (
    OrderRequest,
    OrderSide,
)
from trading_bot.backtesting.portfolio import (
    Portfolio,
)
from trading_bot.market_data.candle import (
    TickerCandle,
)


def create_candle(
    minute: int,
    open_price: str,
    close_price: str,
    symbol: str = "BTC-USDT",
) -> TickerCandle:
    interval_start = datetime(
        2026,
        6,
        27,
        12,
        minute,
        tzinfo=UTC,
    )

    interval_end = interval_start + timedelta(minutes=1)

    open_value = Decimal(open_price)
    close_value = Decimal(close_price)

    high_price = max(
        open_value,
        close_value,
    )

    low_price = min(
        open_value,
        close_value,
    )

    return TickerCandle(
        instrument_id=symbol,
        interval_start=interval_start,
        interval_end=interval_end,
        open_price=open_value,
        high_price=high_price,
        low_price=low_price,
        close_price=close_value,
        first_bid_price=(open_value - Decimal("0.1")),
        last_bid_price=(close_value - Decimal("0.1")),
        first_ask_price=(open_value + Decimal("0.1")),
        last_ask_price=(close_value + Decimal("0.1")),
        minimum_spread=Decimal("0.2"),
        maximum_spread=Decimal("0.2"),
        average_spread=Decimal("0.2"),
        observed_size=Decimal("1"),
        tick_count=1,
        first_exchange_timestamp=(interval_start),
        last_exchange_timestamp=(interval_start),
        first_received_timestamp=(interval_start),
        last_received_timestamp=(interval_start),
    )


def create_engine(
    initial_cash: str = "1000",
    fee_rate: str = "0",
    slippage_basis_points: str = "0",
    liquidate_at_end: bool = False,
) -> tuple[
    BacktestExecutionEngine,
    Portfolio,
]:
    portfolio = Portfolio(initial_cash=Decimal(initial_cash))

    engine = BacktestExecutionEngine(
        portfolio=portfolio,
        fee_model=PercentageFeeModel(rate=Decimal(fee_rate)),
        slippage_basis_points=Decimal(slippage_basis_points),
        liquidate_at_end=(liquidate_at_end),
    )

    return engine, portfolio


def test_order_executes_at_next_candle_open() -> None:
    first = create_candle(
        minute=0,
        open_price="100",
        close_price="105",
    )

    second = create_candle(
        minute=1,
        open_price="110",
        close_price="115",
    )

    order = OrderRequest(
        instrument_id="BTC-USDT",
        side=OrderSide.BUY,
        quantity=Decimal("1"),
        created_at=first.interval_end,
    )

    engine, portfolio = create_engine()

    result = engine.run(
        candles=[first, second],
        orders=[order],
    )

    assert len(result.fills) == 1

    fill = result.fills[0]

    assert fill.timestamp == (second.interval_start)
    assert fill.price == Decimal("110")

    assert portfolio.get_position("BTC-USDT").quantity == Decimal("1")


def test_order_is_not_executed_before_creation() -> None:
    candle = create_candle(
        minute=0,
        open_price="100",
        close_price="105",
    )

    order = OrderRequest(
        instrument_id="BTC-USDT",
        side=OrderSide.BUY,
        quantity=Decimal("1"),
        created_at=(candle.interval_end + timedelta(minutes=5)),
    )

    engine, _ = create_engine()

    result = engine.run(
        candles=[candle],
        orders=[order],
    )

    assert result.fills == []
    assert result.unexecuted_orders == [order]


def test_buy_slippage_increases_price() -> None:
    candle = create_candle(
        minute=0,
        open_price="100",
        close_price="100",
    )

    order = OrderRequest(
        instrument_id="BTC-USDT",
        side=OrderSide.BUY,
        quantity=Decimal("1"),
        created_at=candle.interval_start,
    )

    engine, _ = create_engine(slippage_basis_points="10")

    result = engine.run(
        candles=[candle],
        orders=[order],
    )

    assert result.fills[0].price == Decimal("100.1")


def test_sell_slippage_decreases_price() -> None:
    first = create_candle(
        minute=0,
        open_price="100",
        close_price="100",
    )

    second = create_candle(
        minute=1,
        open_price="120",
        close_price="120",
    )

    buy_order = OrderRequest(
        instrument_id="BTC-USDT",
        side=OrderSide.BUY,
        quantity=Decimal("1"),
        created_at=first.interval_start,
    )

    sell_order = OrderRequest(
        instrument_id="BTC-USDT",
        side=OrderSide.SELL,
        quantity=Decimal("1"),
        created_at=second.interval_start,
    )

    engine, _ = create_engine(slippage_basis_points="10")

    result = engine.run(
        candles=[first, second],
        orders=[
            buy_order,
            sell_order,
        ],
    )

    assert result.fills[1].price == Decimal("119.88")


def test_engine_calculates_fees() -> None:
    candle = create_candle(
        minute=0,
        open_price="100",
        close_price="100",
    )

    order = OrderRequest(
        instrument_id="BTC-USDT",
        side=OrderSide.BUY,
        quantity=Decimal("2"),
        created_at=candle.interval_start,
    )

    engine, portfolio = create_engine(fee_rate="0.001")

    result = engine.run(
        candles=[candle],
        orders=[order],
    )

    assert result.fills[0].fee == Decimal("0.200")

    assert portfolio.fees_paid == Decimal("0.200")


def test_insufficient_cash_rejects_order() -> None:
    candle = create_candle(
        minute=0,
        open_price="100",
        close_price="100",
    )

    order = OrderRequest(
        instrument_id="BTC-USDT",
        side=OrderSide.BUY,
        quantity=Decimal("2"),
        created_at=candle.interval_start,
    )

    engine, _ = create_engine(initial_cash="100")

    result = engine.run(
        candles=[candle],
        orders=[order],
    )

    assert result.fills == []
    assert len(result.rejected_orders) == 1

    assert "Insufficient cash" in result.rejected_orders[0].reason


def test_equity_curve_uses_close_prices() -> None:
    first = create_candle(
        minute=0,
        open_price="100",
        close_price="110",
    )

    second = create_candle(
        minute=1,
        open_price="110",
        close_price="120",
    )

    order = OrderRequest(
        instrument_id="BTC-USDT",
        side=OrderSide.BUY,
        quantity=Decimal("1"),
        created_at=first.interval_start,
    )

    engine, _ = create_engine()

    result = engine.run(
        candles=[first, second],
        orders=[order],
    )

    assert len(result.equity_curve) == 2

    assert result.equity_curve[0].equity == Decimal("1010")

    assert result.equity_curve[1].equity == Decimal("1020")


def test_liquidates_position_at_final_close() -> None:
    candle = create_candle(
        minute=0,
        open_price="100",
        close_price="120",
    )

    order = OrderRequest(
        instrument_id="BTC-USDT",
        side=OrderSide.BUY,
        quantity=Decimal("1"),
        created_at=candle.interval_start,
    )

    engine, portfolio = create_engine(liquidate_at_end=True)

    result = engine.run(
        candles=[candle],
        orders=[order],
    )

    assert len(result.fills) == 2

    assert result.fills[1].side == (OrderSide.SELL)

    assert result.fills[1].price == (Decimal("120"))

    position = portfolio.get_position("BTC-USDT")

    assert position.quantity == Decimal("0")
    assert portfolio.cash == Decimal("1020")
    assert result.final_equity == Decimal("1020")


def test_duplicate_candles_are_rejected() -> None:
    candle = create_candle(
        minute=0,
        open_price="100",
        close_price="100",
    )

    engine, _ = create_engine()

    with pytest.raises(ValueError):
        engine.run(
            candles=[candle, candle],
            orders=[],
        )


def test_empty_candles_leave_orders_unexecuted() -> None:
    timestamp = datetime(
        2026,
        6,
        27,
        12,
        0,
        tzinfo=UTC,
    )

    order = OrderRequest(
        instrument_id="BTC-USDT",
        side=OrderSide.BUY,
        quantity=Decimal("1"),
        created_at=timestamp,
    )

    engine, _ = create_engine()

    result = engine.run(
        candles=[],
        orders=[order],
    )

    assert result.fills == []
    assert result.equity_curve == []
    assert result.unexecuted_orders == [order]
