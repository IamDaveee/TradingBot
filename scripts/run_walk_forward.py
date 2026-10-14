import argparse
from datetime import date
from decimal import Decimal
from pathlib import Path

from trading_bot.backtesting.coverage import (
    CandleCoverageAnalyzer,
)
from trading_bot.backtesting.sweep import (
    create_window_configurations,
)
from trading_bot.backtesting.unified_candle_reader import (
    UnifiedCandleDatasetReader,
)
from trading_bot.backtesting.walk_forward import (
    WalkForwardSettings,
    WalkForwardValidator,
    export_parameter_selection_counts,
    export_walk_forward_folds,
)
from trading_bot.market_data.intervals import (
    SUPPORTED_CANDLE_INTERVALS,
    interval_minutes,
    parse_candle_interval,
)
from trading_bot.settings import load_settings


def parse_date(
    value: str,
) -> date:
    return date.fromisoformat(value)


def parse_integer_list(
    value: str,
) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from error

    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required")

    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("All values must be positive")

    return values


def positive_integer(
    value: str,
) -> int:
    parsed = int(value)

    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero")

    return parsed


def positive_decimal(
    value: str,
) -> Decimal:
    parsed = Decimal(value)

    if not parsed.is_finite() or parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive decimal")

    return parsed


def non_negative_decimal(
    value: str,
) -> Decimal:
    parsed = Decimal(value)

    if not parsed.is_finite() or parsed < 0:
        raise argparse.ArgumentTypeError("Value must be a non-negative decimal")

    return parsed


def candle_interval(
    value: str,
) -> str:
    try:
        return parse_candle_interval(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Run moving-average walk-forward validation"))

    parser.add_argument(
        "--symbol",
        required=True,
    )

    parser.add_argument(
        "--start-date",
        type=parse_date,
        required=True,
    )

    parser.add_argument(
        "--end-date",
        type=parse_date,
        required=True,
    )

    parser.add_argument(
        "--training-days",
        type=positive_integer,
        default=14,
    )

    parser.add_argument(
        "--testing-days",
        type=positive_integer,
        default=7,
    )

    parser.add_argument(
        "--step-days",
        type=positive_integer,
        default=7,
    )

    parser.add_argument(
        "--fast-windows",
        type=parse_integer_list,
        default=[5, 10, 15, 20, 30, 45, 60],
    )

    parser.add_argument(
        "--slow-windows",
        type=parse_integer_list,
        default=[75, 100, 150, 200, 300, 500],
    )

    parser.add_argument(
        "--initial-cash",
        type=positive_decimal,
        default=Decimal("10000"),
    )

    parser.add_argument(
        "--allocation",
        type=positive_decimal,
        default=Decimal("0.5"),
    )

    parser.add_argument(
        "--fee-rate",
        type=non_negative_decimal,
        default=Decimal("0.001"),
    )

    parser.add_argument(
        "--slippage-bps",
        type=non_negative_decimal,
        default=Decimal("5"),
    )

    parser.add_argument(
        "--historical-directory",
        type=Path,
        default=Path("data/processed/historical_candles"),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/backtests/walk_forward"),
    )
    parser.add_argument(
        "--interval",
        type=candle_interval,
        choices=list(SUPPORTED_CANDLE_INTERVALS),
        default="1m",
    )

    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()

    if arguments.end_date < arguments.start_date:
        raise ValueError("end-date must not be before start-date")

    if arguments.allocation > Decimal("1"):
        raise ValueError("allocation must not exceed 1")

    settings = WalkForwardSettings(
        training_days=arguments.training_days,
        testing_days=arguments.testing_days,
        step_days=arguments.step_days,
    )

    configurations = create_window_configurations(
        fast_windows=arguments.fast_windows,
        slow_windows=arguments.slow_windows,
    )

    _, config = load_settings()

    unified_result = UnifiedCandleDatasetReader(
        ticker_dataset_directory=(config.storage.candle_directory),
        historical_dataset_directory=(arguments.historical_directory),
        allowed_symbols=set(config.market_data.symbols),
        allowed_intervals=set(SUPPORTED_CANDLE_INTERVALS),
    ).read(
        interval=arguments.interval,
        symbols=[arguments.symbol],
        start_date=arguments.start_date,
        end_date=arguments.end_date,
    )

    candles = unified_result.candles

    selected_interval_minutes = interval_minutes(arguments.interval)

    coverage = CandleCoverageAnalyzer(
        interval_minutes=(selected_interval_minutes),
        session_gap_threshold_minutes=(selected_interval_minutes * 5),
    ).analyze(candles)

    print(f"Interval: {arguments.interval} ({selected_interval_minutes} minutes)")

    print(f"Historical source candles: {unified_result.historical_candle_count}")
    print(f"Ticker-derived source candles: {unified_result.ticker_candle_count}")
    print(f"Ticker-over-historical overlaps: {unified_result.overlap_count}")
    print(f"Candles: {len(candles)}")
    print(f"Missing intervals: {coverage.total_missing_intervals}")
    print(f"Session gaps: {coverage.total_session_gaps}")
    print(f"Duplicate intervals: {coverage.total_duplicates}")

    if coverage.total_duplicates > 0:
        raise ValueError("Walk-forward validation cannot run with duplicate candle intervals")

    if coverage.total_missing_intervals > 0:
        raise ValueError("Walk-forward validation cannot run with missing candle intervals")

    if coverage.total_session_gaps > 0:
        raise ValueError("Walk-forward validation cannot run with session gaps")

    validator = WalkForwardValidator(
        settings=settings,
        initial_cash=arguments.initial_cash,
        allocation_fraction=(arguments.allocation),
        fee_rate=arguments.fee_rate,
        slippage_basis_points=(arguments.slippage_bps),
        liquidate_at_end=True,
    )

    result = validator.run(
        candles=candles,
        configurations=configurations,
    )

    summary = result.summary

    print("\nWalk-forward folds:")

    for fold in result.folds:
        profit_factor = (
            str(fold.testing_profit_factor)
            if fold.testing_profit_factor is not None
            else "infinite"
        )

        print(
            f"  Fold {fold.fold_number}: "
            f"train={fold.training_start}"
            f"..{fold.training_end} "
            f"test={fold.testing_start}"
            f"..{fold.testing_end}"
        )

        print(
            f"    selected="
            f"{fold.selected_fast_window}/"
            f"{fold.selected_slow_window} "
            f"train_return="
            f"{fold.training_return_percentage}% "
            f"test_return="
            f"{fold.testing_return_percentage}% "
            f"test_drawdown="
            f"{fold.testing_maximum_drawdown_percentage}% "
            f"trades="
            f"{fold.testing_completed_trade_count} "
            f"profit_factor={profit_factor}"
        )

    print("\nAggregate out-of-sample result:")
    print(f"  Initial equity: {summary.initial_equity}")
    print(f"  Final equity: {summary.final_equity}")
    print(f"  Absolute return: {summary.absolute_return}")
    print(f"  Return percentage: {summary.return_percentage}%")
    print(f"  Maximum drawdown: {summary.maximum_drawdown}")
    print(f"  Maximum drawdown percentage: {summary.maximum_drawdown_percentage}%")
    print(f"  Folds: {summary.fold_count}")
    print(f"  Profitable folds: {summary.profitable_fold_count}")
    print(f"  Losing folds: {summary.losing_fold_count}")
    print(f"  Completed trades: {summary.total_completed_trades}")
    print(f"  Fees paid: {summary.total_fees}")

    if (
        summary.most_selected_fast_window is not None
        and summary.most_selected_slow_window is not None
    ):
        print(
            "  Most selected parameters: "
            f"{summary.most_selected_fast_window}/"
            f"{summary.most_selected_slow_window} "
            f"({summary.most_selected_parameter_count} folds)"
        )

    output_directory = arguments.output_directory

    export_walk_forward_folds(
        result=result,
        path=(output_directory / "walk_forward_folds.csv"),
    )

    export_parameter_selection_counts(
        result=result,
        path=(output_directory / "parameter_selection_counts.csv"),
    )

    print(f"\nExported walk-forward results to: {output_directory}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
