from collections.abc import Iterable

from trading_bot.backtesting.execution import (
    BacktestExecutionEngine,
    BacktestResult,
)
from trading_bot.backtesting.strategy import (
    Strategy,
)
from trading_bot.market_data.candle import (
    TickerCandle,
)


class StrategyBacktestRunner:
    def __init__(
        self,
        execution_engine: BacktestExecutionEngine,
        strategy: Strategy,
    ) -> None:
        self._execution_engine = execution_engine

        self._strategy = strategy

    def run(
        self,
        candles: Iterable[TickerCandle],
    ) -> BacktestResult:
        return self._execution_engine.run_strategy(
            candles=candles,
            strategy=self._strategy,
        )
