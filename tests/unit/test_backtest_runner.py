from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal

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
from trading_bot.backtesting.runner import (
    StrategyBacktestRunner,
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


def candle(
    minute: int,
    price: str,
) -> TickerCandle:
    start = datetime(
        2026,
        6,
        27,
        12,
        minute,
        tzinfo=UTC,
    )

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


def create_runner(
    strategy: MovingAverageCrossoverStrategy,
    initial_cash: str = "1000",
    fee_rate: str = "0",
) -> tuple[
    StrategyBacktestRunner,
    Portfolio,
]:
    portfolio = Portfolio(initial_cash=Decimal(initial_cash))

    engine = BacktestExecutionEngine(
        portfolio=portfolio,
        fee_model=PercentageFeeModel(rate=Decimal(fee_rate)),
    )

    runner = StrategyBacktestRunner(
        execution_engine=engine,
        strategy=strategy,
    )

    return runner, portfolio


def test_runner_executes_buy_and_sell_signals() -> None:
    candles = [
        candle(0, "100"),
        candle(1, "99"),
        candle(2, "98"),
        candle(3, "110"),
        candle(4, "111"),
        candle(5, "90"),
        candle(6, "89"),
    ]

    strategy = MovingAverageCrossoverStrategy(
        fast_window=2,
        slow_window=3,
        allocation_fraction=Decimal("0.5"),
    )

    runner, portfolio = create_runner(strategy)

    result = runner.run(candles)

    assert len(result.fills) == 2

    buy_fill = result.fills[0]
    sell_fill = result.fills[1]

    assert buy_fill.timestamp == (candles[4].interval_start)

    assert sell_fill.timestamp == (candles[6].interval_start)

    assert portfolio.get_position("BTC-USDT").quantity == Decimal("0")


@dataclass(slots=True)
class ContextRecordingStrategy:
    contexts: list[StrategyContext] = field(default_factory=list)

    def on_candle(
        self,
        candle: TickerCandle,
        context: StrategyContext,
    ) -> list[OrderRequest]:
        self.contexts.append(context)
        return []


def test_strategy_receives_current_portfolio_state() -> None:
    candles = [
        candle(0, "100"),
        candle(1, "110"),
    ]

    portfolio = Portfolio(initial_cash=Decimal("1000"))

    engine = BacktestExecutionEngine(
        portfolio=portfolio,
        fee_model=PercentageFeeModel(rate=Decimal("0")),
    )

    strategy = ContextRecordingStrategy()

    runner = StrategyBacktestRunner(
        execution_engine=engine,
        strategy=strategy,
    )

    runner.run(candles)

    assert len(strategy.contexts) == 2

    assert strategy.contexts[0].cash == Decimal("1000")

    assert strategy.contexts[1].cash == Decimal("1000")


def test_final_signal_remains_unexecuted() -> None:
    candles = [
        candle(0, "100"),
        candle(1, "99"),
        candle(2, "98"),
        candle(3, "110"),
    ]

    strategy = MovingAverageCrossoverStrategy(
        fast_window=2,
        slow_window=3,
        allocation_fraction=Decimal("0.5"),
    )

    runner, _ = create_runner(strategy)

    result = runner.run(candles)

    assert result.fills == []

    assert len(result.unexecuted_orders) == 1

    assert result.unexecuted_orders[0].created_at == candles[-1].interval_end


@dataclass(slots=True)
class BuyOnceRecordingStrategy:
    contexts: list[StrategyContext] = field(default_factory=list)

    def on_candle(
        self,
        candle: TickerCandle,
        context: StrategyContext,
    ) -> list[OrderRequest]:
        self.contexts.append(context)

        if len(self.contexts) == 1:
            return [
                OrderRequest(
                    instrument_id=(candle.instrument_id),
                    side=OrderSide.BUY,
                    quantity=Decimal("1"),
                    created_at=(candle.interval_end),
                )
            ]

        return []


def test_context_reflects_executed_fill() -> None:
    candles = [
        candle(0, "100"),
        candle(1, "110"),
        candle(2, "120"),
    ]

    portfolio = Portfolio(initial_cash=Decimal("1000"))

    engine = BacktestExecutionEngine(
        portfolio=portfolio,
        fee_model=PercentageFeeModel(rate=Decimal("0")),
    )

    strategy = BuyOnceRecordingStrategy()

    runner = StrategyBacktestRunner(
        execution_engine=engine,
        strategy=strategy,
    )

    result = runner.run(candles)

    assert len(result.fills) == 1

    assert result.fills[0].price == (Decimal("110"))

    second_context = strategy.contexts[1]

    assert second_context.cash == (Decimal("890"))

    assert second_context.position.quantity == Decimal("1")

    assert second_context.equity == (Decimal("1000"))
