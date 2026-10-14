import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from trading_bot.backtesting.execution import (
    BacktestExecutionEngine,
)
from trading_bot.backtesting.fees import (
    PercentageFeeModel,
)
from trading_bot.backtesting.metrics import (
    calculate_backtest_metrics,
)
from trading_bot.backtesting.portfolio import (
    Portfolio,
)
from trading_bot.backtesting.runner import (
    StrategyBacktestRunner,
)
from trading_bot.backtesting.trades import (
    TradeAnalyzer,
    calculate_trade_statistics,
)
from trading_bot.market_data.candle import (
    TickerCandle,
)
from trading_bot.strategies.moving_average_crossover import (
    MovingAverageCrossoverStrategy,
)


@dataclass(frozen=True, slots=True)
class SweepConfiguration:
    fast_window: int
    slow_window: int


@dataclass(frozen=True, slots=True)
class SweepResult:
    fast_window: int
    slow_window: int
    final_equity: Decimal
    absolute_return: Decimal
    return_percentage: Decimal
    maximum_drawdown: Decimal
    maximum_drawdown_percentage: Decimal
    fill_count: int
    completed_trade_count: int
    win_rate_percentage: Decimal
    profit_factor: Decimal | None
    total_fees: Decimal
    rejected_order_count: int
    unexecuted_order_count: int


class MovingAverageParameterSweep:
    def __init__(
        self,
        initial_cash: Decimal,
        allocation_fraction: Decimal,
        fee_rate: Decimal,
        slippage_basis_points: Decimal,
        liquidate_at_end: bool = True,
    ) -> None:
        if initial_cash <= 0:
            raise ValueError("initial_cash must be greater than zero")

        if not (Decimal("0") < allocation_fraction <= Decimal("1")):
            raise ValueError("allocation_fraction must be within (0, 1]")

        self._initial_cash = initial_cash
        self._allocation_fraction = allocation_fraction
        self._fee_rate = fee_rate
        self._slippage_basis_points = slippage_basis_points
        self._liquidate_at_end = liquidate_at_end

    def run(
        self,
        candles: list[TickerCandle],
        configurations: list[SweepConfiguration],
    ) -> list[SweepResult]:
        if not candles:
            raise ValueError("At least one candle is required")

        results: list[SweepResult] = []

        for configuration in configurations:
            if configuration.fast_window <= 0:
                raise ValueError("fast_window must be greater than zero")

            if configuration.slow_window <= 0:
                raise ValueError("slow_window must be greater than zero")

            if configuration.fast_window >= configuration.slow_window:
                raise ValueError("fast_window must be below slow_window")

            portfolio = Portfolio(initial_cash=self._initial_cash)

            engine = BacktestExecutionEngine(
                portfolio=portfolio,
                fee_model=PercentageFeeModel(rate=self._fee_rate),
                slippage_basis_points=(self._slippage_basis_points),
                liquidate_at_end=(self._liquidate_at_end),
            )

            strategy = MovingAverageCrossoverStrategy(
                fast_window=(configuration.fast_window),
                slow_window=(configuration.slow_window),
                allocation_fraction=(self._allocation_fraction),
            )

            runner = StrategyBacktestRunner(
                execution_engine=engine,
                strategy=strategy,
            )

            backtest_result = runner.run(candles)

            metrics = calculate_backtest_metrics(
                result=backtest_result,
                initial_equity=(self._initial_cash),
            )

            completed_trades, _ = TradeAnalyzer().analyze(backtest_result.fills)

            statistics = calculate_trade_statistics(completed_trades)

            results.append(
                SweepResult(
                    fast_window=(configuration.fast_window),
                    slow_window=(configuration.slow_window),
                    final_equity=(metrics.final_equity),
                    absolute_return=(metrics.absolute_return),
                    return_percentage=(metrics.return_percentage),
                    maximum_drawdown=(metrics.maximum_drawdown),
                    maximum_drawdown_percentage=(metrics.maximum_drawdown_percentage),
                    fill_count=metrics.fill_count,
                    completed_trade_count=(statistics.completed_trade_count),
                    win_rate_percentage=(statistics.win_rate_percentage),
                    profit_factor=(statistics.profit_factor),
                    total_fees=(metrics.total_fees),
                    rejected_order_count=(metrics.rejected_order_count),
                    unexecuted_order_count=(metrics.unexecuted_order_count),
                )
            )

        return sorted(
            results,
            key=lambda result: (
                result.return_percentage,
                -result.maximum_drawdown_percentage,
                result.completed_trade_count,
            ),
            reverse=True,
        )


def create_window_configurations(
    fast_windows: list[int],
    slow_windows: list[int],
) -> list[SweepConfiguration]:
    configurations = [
        SweepConfiguration(
            fast_window=fast,
            slow_window=slow,
        )
        for fast in fast_windows
        for slow in slow_windows
        if fast < slow
    ]

    if not configurations:
        raise ValueError("No valid fast/slow window combinations were produced")

    return configurations


def export_sweep_results(
    results: list[SweepResult],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "rank",
                "fast_window",
                "slow_window",
                "final_equity",
                "absolute_return",
                "return_percentage",
                "maximum_drawdown",
                "maximum_drawdown_percentage",
                "fill_count",
                "completed_trade_count",
                "win_rate_percentage",
                "profit_factor",
                "total_fees",
                "rejected_order_count",
                "unexecuted_order_count",
            ]
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):
            writer.writerow(
                [
                    rank,
                    result.fast_window,
                    result.slow_window,
                    str(result.final_equity),
                    str(result.absolute_return),
                    str(result.return_percentage),
                    str(result.maximum_drawdown),
                    str(result.maximum_drawdown_percentage),
                    result.fill_count,
                    result.completed_trade_count,
                    str(result.win_rate_percentage),
                    (str(result.profit_factor) if result.profit_factor is not None else "infinite"),
                    str(result.total_fees),
                    result.rejected_order_count,
                    result.unexecuted_order_count,
                ]
            )
