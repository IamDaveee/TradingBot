import csv
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path

from trading_bot.backtesting.execution import (
    BacktestExecutionEngine,
    EquitySnapshot,
)
from trading_bot.backtesting.fees import (
    PercentageFeeModel,
)
from trading_bot.backtesting.metrics import (
    BacktestMetrics,
    calculate_backtest_metrics,
)
from trading_bot.backtesting.portfolio import (
    Portfolio,
)
from trading_bot.backtesting.runner import (
    StrategyBacktestRunner,
)
from trading_bot.backtesting.sweep import (
    MovingAverageParameterSweep,
    SweepConfiguration,
    SweepResult,
)
from trading_bot.backtesting.trades import (
    TradeAnalyzer,
    TradeStatistics,
    calculate_trade_statistics,
)
from trading_bot.market_data.candle import (
    TickerCandle,
)
from trading_bot.strategies.moving_average_crossover import (
    MovingAverageCrossoverStrategy,
)


@dataclass(frozen=True, slots=True)
class WalkForwardSettings:
    training_days: int
    testing_days: int
    step_days: int

    def __post_init__(self) -> None:
        if self.training_days <= 0:
            raise ValueError("training_days must be greater than zero")

        if self.testing_days <= 0:
            raise ValueError("testing_days must be greater than zero")

        if self.step_days <= 0:
            raise ValueError("step_days must be greater than zero")

        if self.step_days < self.testing_days:
            raise ValueError(
                "step_days must be at least testing_days to prevent overlapping test periods"
            )


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    fold_number: int

    training_start: date
    training_end: date

    testing_start: date
    testing_end: date

    training_candle_count: int
    testing_candle_count: int

    selected_fast_window: int
    selected_slow_window: int

    training_return_percentage: Decimal
    training_maximum_drawdown_percentage: Decimal
    training_completed_trade_count: int

    testing_initial_equity: Decimal
    testing_final_equity: Decimal
    testing_absolute_return: Decimal
    testing_return_percentage: Decimal
    testing_maximum_drawdown: Decimal
    testing_maximum_drawdown_percentage: Decimal

    testing_fill_count: int
    testing_completed_trade_count: int
    testing_win_rate_percentage: Decimal
    testing_profit_factor: Decimal | None
    testing_total_fees: Decimal
    testing_rejected_order_count: int
    testing_unexecuted_order_count: int


@dataclass(frozen=True, slots=True)
class WalkForwardSummary:
    initial_equity: Decimal
    final_equity: Decimal
    absolute_return: Decimal
    return_percentage: Decimal

    maximum_drawdown: Decimal
    maximum_drawdown_percentage: Decimal

    fold_count: int
    profitable_fold_count: int
    losing_fold_count: int

    total_completed_trades: int
    total_fees: Decimal

    most_selected_fast_window: int | None
    most_selected_slow_window: int | None
    most_selected_parameter_count: int

    parameter_selection_counts: dict[
        tuple[int, int],
        int,
    ]


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    folds: list[WalkForwardFold]
    summary: WalkForwardSummary
    out_of_sample_equity_curve: list[EquitySnapshot]


class WalkForwardValidator:
    def __init__(
        self,
        settings: WalkForwardSettings,
        initial_cash: Decimal,
        allocation_fraction: Decimal,
        fee_rate: Decimal,
        slippage_basis_points: Decimal,
        liquidate_at_end: bool = True,
    ) -> None:
        if not initial_cash.is_finite() or initial_cash <= 0:
            raise ValueError("initial_cash must be a positive finite Decimal")

        if (
            not allocation_fraction.is_finite()
            or allocation_fraction <= 0
            or allocation_fraction > 1
        ):
            raise ValueError("allocation_fraction must be within (0, 1]")

        if not fee_rate.is_finite() or fee_rate < 0 or fee_rate >= 1:
            raise ValueError("fee_rate must be within [0, 1)")

        if (
            not slippage_basis_points.is_finite()
            or slippage_basis_points < 0
            or slippage_basis_points >= 10000
        ):
            raise ValueError("slippage_basis_points must be within [0, 10000)")

        self._settings = settings
        self._initial_cash = initial_cash
        self._allocation_fraction = allocation_fraction
        self._fee_rate = fee_rate
        self._slippage_basis_points = slippage_basis_points
        self._liquidate_at_end = liquidate_at_end

    def run(
        self,
        candles: list[TickerCandle],
        configurations: list[SweepConfiguration],
    ) -> WalkForwardResult:
        self._validate_inputs(
            candles=candles,
            configurations=configurations,
        )

        sorted_candles = sorted(
            candles,
            key=lambda candle: (
                candle.interval_start,
                candle.instrument_id,
            ),
        )

        fold_ranges = self._create_fold_ranges(sorted_candles)

        if not fold_ranges:
            raise ValueError(
                "The dataset is too short for the configured training and testing windows"
            )

        folds: list[WalkForwardFold] = []

        combined_equity_curve: list[EquitySnapshot] = []

        current_cash = self._initial_cash

        maximum_slow_window = max(configuration.slow_window for configuration in configurations)

        for fold_number, (
            training_start,
            training_end_exclusive,
            testing_start,
            testing_end_exclusive,
        ) in enumerate(
            fold_ranges,
            start=1,
        ):
            training_candles = self._filter_candles(
                candles=sorted_candles,
                start=training_start,
                end_exclusive=(training_end_exclusive),
            )

            testing_candles = self._filter_candles(
                candles=sorted_candles,
                start=testing_start,
                end_exclusive=(testing_end_exclusive),
            )

            minimum_training_candles = maximum_slow_window + 1

            if len(training_candles) < minimum_training_candles:
                raise ValueError(
                    f"Fold {fold_number} has insufficient "
                    "training candles: "
                    f"required={minimum_training_candles}, "
                    f"available={len(training_candles)}"
                )

            if len(testing_candles) < 2:
                raise ValueError(
                    f"Fold {fold_number} has insufficient "
                    "testing candles: "
                    f"available={len(testing_candles)}"
                )

            training_results = self._run_training_sweep(
                candles=training_candles,
                configurations=configurations,
                initial_cash=current_cash,
            )

            selected = training_results[0]

            (
                testing_metrics,
                testing_statistics,
                testing_equity_curve,
            ) = self._run_out_of_sample_test(
                candles=testing_candles,
                selected=selected,
                initial_cash=current_cash,
            )

            folds.append(
                WalkForwardFold(
                    fold_number=fold_number,
                    training_start=(training_start.date()),
                    training_end=(training_end_exclusive.date() - timedelta(days=1)),
                    testing_start=(testing_start.date()),
                    testing_end=(testing_end_exclusive.date() - timedelta(days=1)),
                    training_candle_count=len(training_candles),
                    testing_candle_count=len(testing_candles),
                    selected_fast_window=(selected.fast_window),
                    selected_slow_window=(selected.slow_window),
                    training_return_percentage=(selected.return_percentage),
                    training_maximum_drawdown_percentage=(selected.maximum_drawdown_percentage),
                    training_completed_trade_count=(selected.completed_trade_count),
                    testing_initial_equity=(current_cash),
                    testing_final_equity=(testing_metrics.final_equity),
                    testing_absolute_return=(testing_metrics.absolute_return),
                    testing_return_percentage=(testing_metrics.return_percentage),
                    testing_maximum_drawdown=(testing_metrics.maximum_drawdown),
                    testing_maximum_drawdown_percentage=(
                        testing_metrics.maximum_drawdown_percentage
                    ),
                    testing_fill_count=(testing_metrics.fill_count),
                    testing_completed_trade_count=(testing_statistics.completed_trade_count),
                    testing_win_rate_percentage=(testing_statistics.win_rate_percentage),
                    testing_profit_factor=(testing_statistics.profit_factor),
                    testing_total_fees=(testing_metrics.total_fees),
                    testing_rejected_order_count=(testing_metrics.rejected_order_count),
                    testing_unexecuted_order_count=(testing_metrics.unexecuted_order_count),
                )
            )

            combined_equity_curve.extend(testing_equity_curve)

            current_cash = testing_metrics.final_equity

        summary = self._calculate_summary(
            folds=folds,
            equity_curve=combined_equity_curve,
        )

        return WalkForwardResult(
            folds=folds,
            summary=summary,
            out_of_sample_equity_curve=(combined_equity_curve),
        )

    def _run_training_sweep(
        self,
        candles: list[TickerCandle],
        configurations: list[SweepConfiguration],
        initial_cash: Decimal,
    ) -> list[SweepResult]:
        sweep = MovingAverageParameterSweep(
            initial_cash=initial_cash,
            allocation_fraction=(self._allocation_fraction),
            fee_rate=self._fee_rate,
            slippage_basis_points=(self._slippage_basis_points),
            liquidate_at_end=(self._liquidate_at_end),
        )

        return sweep.run(
            candles=candles,
            configurations=configurations,
        )

    def _run_out_of_sample_test(
        self,
        candles: list[TickerCandle],
        selected: SweepResult,
        initial_cash: Decimal,
    ) -> tuple[
        BacktestMetrics,
        TradeStatistics,
        list[EquitySnapshot],
    ]:
        portfolio = Portfolio(initial_cash=initial_cash)

        engine = BacktestExecutionEngine(
            portfolio=portfolio,
            fee_model=PercentageFeeModel(rate=self._fee_rate),
            slippage_basis_points=(self._slippage_basis_points),
            liquidate_at_end=(self._liquidate_at_end),
        )

        strategy = MovingAverageCrossoverStrategy(
            fast_window=(selected.fast_window),
            slow_window=(selected.slow_window),
            allocation_fraction=(self._allocation_fraction),
        )

        runner = StrategyBacktestRunner(
            execution_engine=engine,
            strategy=strategy,
        )

        backtest_result = runner.run(candles)

        metrics = calculate_backtest_metrics(
            result=backtest_result,
            initial_equity=initial_cash,
        )

        completed_trades, _ = TradeAnalyzer().analyze(backtest_result.fills)

        statistics = calculate_trade_statistics(completed_trades)

        return (
            metrics,
            statistics,
            backtest_result.equity_curve,
        )

    def _create_fold_ranges(
        self,
        candles: list[TickerCandle],
    ) -> list[
        tuple[
            datetime,
            datetime,
            datetime,
            datetime,
        ]
    ]:
        first_day = min(candle.interval_start for candle in candles).astimezone(UTC).date()

        last_day = max(candle.interval_start for candle in candles).astimezone(UTC).date()

        dataset_end_exclusive = self._utc_midnight(last_day + timedelta(days=1))

        training_start = self._utc_midnight(first_day)

        ranges: list[
            tuple[
                datetime,
                datetime,
                datetime,
                datetime,
            ]
        ] = []

        while True:
            training_end_exclusive = training_start + timedelta(days=(self._settings.training_days))

            testing_start = training_end_exclusive

            testing_end_exclusive = testing_start + timedelta(days=(self._settings.testing_days))

            if testing_end_exclusive > dataset_end_exclusive:
                break

            ranges.append(
                (
                    training_start,
                    training_end_exclusive,
                    testing_start,
                    testing_end_exclusive,
                )
            )

            training_start += timedelta(days=self._settings.step_days)

        return ranges

    @staticmethod
    def _filter_candles(
        candles: list[TickerCandle],
        start: datetime,
        end_exclusive: datetime,
    ) -> list[TickerCandle]:
        return [candle for candle in candles if (start <= candle.interval_start < end_exclusive)]

    def _calculate_summary(
        self,
        folds: list[WalkForwardFold],
        equity_curve: list[EquitySnapshot],
    ) -> WalkForwardSummary:
        final_equity = folds[-1].testing_final_equity

        absolute_return = final_equity - self._initial_cash

        return_percentage = absolute_return / self._initial_cash * Decimal("100")

        (
            maximum_drawdown,
            maximum_drawdown_percentage,
        ) = self._maximum_drawdown(equity_curve)

        profitable_fold_count = sum(1 for fold in folds if fold.testing_absolute_return > 0)

        losing_fold_count = sum(1 for fold in folds if fold.testing_absolute_return < 0)

        parameter_counts = Counter(
            (
                fold.selected_fast_window,
                fold.selected_slow_window,
            )
            for fold in folds
        )

        if parameter_counts:
            (
                most_selected_parameters,
                most_selected_count,
            ) = parameter_counts.most_common(1)[0]

            most_selected_fast = most_selected_parameters[0]

            most_selected_slow = most_selected_parameters[1]
        else:
            most_selected_fast = None
            most_selected_slow = None
            most_selected_count = 0

        return WalkForwardSummary(
            initial_equity=self._initial_cash,
            final_equity=final_equity,
            absolute_return=absolute_return,
            return_percentage=return_percentage,
            maximum_drawdown=maximum_drawdown,
            maximum_drawdown_percentage=(maximum_drawdown_percentage),
            fold_count=len(folds),
            profitable_fold_count=(profitable_fold_count),
            losing_fold_count=(losing_fold_count),
            total_completed_trades=sum(fold.testing_completed_trade_count for fold in folds),
            total_fees=sum(
                (fold.testing_total_fees for fold in folds),
                start=Decimal("0"),
            ),
            most_selected_fast_window=(most_selected_fast),
            most_selected_slow_window=(most_selected_slow),
            most_selected_parameter_count=(most_selected_count),
            parameter_selection_counts=dict(parameter_counts),
        )

    @staticmethod
    def _maximum_drawdown(
        equity_curve: list[EquitySnapshot],
    ) -> tuple[Decimal, Decimal]:
        if not equity_curve:
            return (
                Decimal("0"),
                Decimal("0"),
            )

        peak = equity_curve[0].equity
        maximum_drawdown = Decimal("0")
        maximum_percentage = Decimal("0")

        for snapshot in equity_curve:
            if snapshot.equity > peak:
                peak = snapshot.equity
                continue

            drawdown = peak - snapshot.equity

            if drawdown > maximum_drawdown:
                maximum_drawdown = drawdown

            if peak > 0:
                percentage = drawdown / peak * Decimal("100")

                if percentage > maximum_percentage:
                    maximum_percentage = percentage

        return (
            maximum_drawdown,
            maximum_percentage,
        )

    @staticmethod
    def _validate_inputs(
        candles: list[TickerCandle],
        configurations: list[SweepConfiguration],
    ) -> None:
        if not candles:
            raise ValueError("At least one candle is required")

        if not configurations:
            raise ValueError("At least one sweep configuration is required")

        symbols = {candle.instrument_id for candle in candles}

        if len(symbols) != 1:
            raise ValueError("Walk-forward validation currently supports exactly one instrument")

    @staticmethod
    def _utc_midnight(
        value: date,
    ) -> datetime:
        return datetime.combine(
            value,
            time.min,
            tzinfo=UTC,
        )


def export_walk_forward_folds(
    result: WalkForwardResult,
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
                "fold_number",
                "training_start",
                "training_end",
                "testing_start",
                "testing_end",
                "training_candle_count",
                "testing_candle_count",
                "selected_fast_window",
                "selected_slow_window",
                "training_return_percentage",
                "training_maximum_drawdown_percentage",
                "training_completed_trade_count",
                "testing_initial_equity",
                "testing_final_equity",
                "testing_absolute_return",
                "testing_return_percentage",
                "testing_maximum_drawdown",
                "testing_maximum_drawdown_percentage",
                "testing_fill_count",
                "testing_completed_trade_count",
                "testing_win_rate_percentage",
                "testing_profit_factor",
                "testing_total_fees",
                "testing_rejected_order_count",
                "testing_unexecuted_order_count",
            ]
        )

        for fold in result.folds:
            writer.writerow(
                [
                    fold.fold_number,
                    fold.training_start.isoformat(),
                    fold.training_end.isoformat(),
                    fold.testing_start.isoformat(),
                    fold.testing_end.isoformat(),
                    fold.training_candle_count,
                    fold.testing_candle_count,
                    fold.selected_fast_window,
                    fold.selected_slow_window,
                    str(fold.training_return_percentage),
                    str(fold.training_maximum_drawdown_percentage),
                    fold.training_completed_trade_count,
                    str(fold.testing_initial_equity),
                    str(fold.testing_final_equity),
                    str(fold.testing_absolute_return),
                    str(fold.testing_return_percentage),
                    str(fold.testing_maximum_drawdown),
                    str(fold.testing_maximum_drawdown_percentage),
                    fold.testing_fill_count,
                    fold.testing_completed_trade_count,
                    str(fold.testing_win_rate_percentage),
                    (
                        str(fold.testing_profit_factor)
                        if fold.testing_profit_factor is not None
                        else "infinite"
                    ),
                    str(fold.testing_total_fees),
                    fold.testing_rejected_order_count,
                    fold.testing_unexecuted_order_count,
                ]
            )


def export_parameter_selection_counts(
    result: WalkForwardResult,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sorted_counts = sorted(
        result.summary.parameter_selection_counts.items(),
        key=lambda item: (
            item[1],
            item[0][0],
            item[0][1],
        ),
        reverse=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "fast_window",
                "slow_window",
                "selection_count",
            ]
        )

        for (
            fast_window,
            slow_window,
        ), count in sorted_counts:
            writer.writerow(
                [
                    fast_window,
                    slow_window,
                    count,
                ]
            )
