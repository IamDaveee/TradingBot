from dataclasses import dataclass
from decimal import Decimal

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
from trading_bot.strategies.atr_filtered_ma_crossover import (
    AtrFilteredMovingAverageCrossoverStrategy,
)


@dataclass(frozen=True, slots=True)
class AtrSweepConfiguration:
    fast_window: int
    slow_window: int
    atr_period: int
    atr_multiplier: Decimal
    minimum_holding_candles: int = 0
    cooldown_candles: int = 0


@dataclass(frozen=True, slots=True)
class AtrSweepResult:
    fast_window: int
    slow_window: int
    atr_period: int
    atr_multiplier: Decimal
    minimum_holding_candles: int
    cooldown_candles: int

    return_percentage: Decimal
    maximum_drawdown_percentage: Decimal
    completed_trade_count: int
    win_rate_percentage: Decimal
    profit_factor: Decimal | None
    total_fees: Decimal


class AtrParameterSweep:
    def __init__(
        self,
        initial_cash: Decimal,
        allocation_fraction: Decimal,
        fee_rate: Decimal,
        slippage_basis_points: Decimal,
        estimated_round_trip_cost_rate: Decimal,
        liquidate_at_end: bool = True,
    ) -> None:
        self._initial_cash = initial_cash
        self._allocation_fraction = allocation_fraction
        self._fee_rate = fee_rate

        self._slippage_basis_points = slippage_basis_points

        self._estimated_round_trip_cost_rate = estimated_round_trip_cost_rate

        self._liquidate_at_end = liquidate_at_end

    def run(
        self,
        candles: list[TickerCandle],
        configurations: list[AtrSweepConfiguration],
    ) -> list[AtrSweepResult]:
        results: list[AtrSweepResult] = []

        for configuration in configurations:
            portfolio = Portfolio(initial_cash=self._initial_cash)

            engine = BacktestExecutionEngine(
                portfolio=portfolio,
                fee_model=PercentageFeeModel(rate=self._fee_rate),
                slippage_basis_points=(self._slippage_basis_points),
                liquidate_at_end=(self._liquidate_at_end),
            )

            strategy = AtrFilteredMovingAverageCrossoverStrategy(
                fast_window=(configuration.fast_window),
                slow_window=(configuration.slow_window),
                atr_period=(configuration.atr_period),
                atr_multiplier=(configuration.atr_multiplier),
                allocation_fraction=(self._allocation_fraction),
                estimated_round_trip_cost_rate=(self._estimated_round_trip_cost_rate),
                minimum_holding_candles=(configuration.minimum_holding_candles),
                cooldown_candles=(configuration.cooldown_candles),
            )

            backtest_result = StrategyBacktestRunner(
                execution_engine=engine,
                strategy=strategy,
            ).run(candles)

            metrics = calculate_backtest_metrics(
                result=backtest_result,
                initial_equity=(self._initial_cash),
            )

            completed_trades, _ = TradeAnalyzer().analyze(backtest_result.fills)

            statistics = calculate_trade_statistics(completed_trades)

            results.append(
                AtrSweepResult(
                    fast_window=(configuration.fast_window),
                    slow_window=(configuration.slow_window),
                    atr_period=(configuration.atr_period),
                    atr_multiplier=(configuration.atr_multiplier),
                    minimum_holding_candles=(configuration.minimum_holding_candles),
                    cooldown_candles=(configuration.cooldown_candles),
                    return_percentage=(metrics.return_percentage),
                    maximum_drawdown_percentage=(metrics.maximum_drawdown_percentage),
                    completed_trade_count=(statistics.completed_trade_count),
                    win_rate_percentage=(statistics.win_rate_percentage),
                    profit_factor=(statistics.profit_factor),
                    total_fees=(metrics.total_fees),
                )
            )

        return sorted(
            results,
            key=lambda result: (
                result.return_percentage,
                -result.maximum_drawdown_percentage,
                result.profit_factor if result.profit_factor is not None else Decimal("Infinity"),
            ),
            reverse=True,
        )
