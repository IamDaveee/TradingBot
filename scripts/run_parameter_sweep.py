import argparse
from datetime import date
from decimal import Decimal
from pathlib import Path

from trading_bot.backtesting.baseline import (
    calculate_buy_and_hold,
)
from trading_bot.backtesting.coverage import (
    CandleCoverageAnalyzer,
    validate_minimum_history,
)
from trading_bot.backtesting.sweep import (
    MovingAverageParameterSweep,
    create_window_configurations,
    export_sweep_results,
)
from trading_bot.backtesting.unified_candle_reader import (
    UnifiedCandleDatasetReader,
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
        raise argparse.ArgumentTypeError("Window values must be positive")

    return values


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
    parser = argparse.ArgumentParser(description=("Run a moving-average parameter sweep"))

    parser.add_argument(
        "--symbol",
        required=True,
    )

    parser.add_argument(
        "--start-date",
        type=parse_date,
    )

    parser.add_argument(
        "--end-date",
        type=parse_date,
    )

    parser.add_argument(
        "--fast-windows",
        type=parse_integer_list,
        default=[2, 3, 5, 8],
    )

    parser.add_argument(
        "--slow-windows",
        type=parse_integer_list,
        default=[10, 15, 20, 30],
    )

    parser.add_argument(
        "--initial-cash",
        type=positive_decimal,
        default=Decimal("10000"),
    )

    parser.add_argument(
        "--allocation",
        type=positive_decimal,
        default=Decimal("0.95"),
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
        "--session-gap-threshold",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--allow-session-gaps",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/backtests/parameter_sweep.csv"),
    )

    parser.add_argument(
        "--top",
        type=int,
        default=10,
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

    if arguments.allocation > Decimal("1"):
        raise ValueError("allocation must not exceed 1")

    if arguments.top <= 0:
        raise ValueError("top must be greater than zero")

    configurations = create_window_configurations(
        fast_windows=(arguments.fast_windows),
        slow_windows=(arguments.slow_windows),
    )

    maximum_slow_window = max(configuration.slow_window for configuration in configurations)

    _, config = load_settings()

    unified_result = UnifiedCandleDatasetReader(
        ticker_dataset_directory=(config.storage.candle_directory),
        historical_dataset_directory=Path("data/processed/historical_candles"),
        allowed_symbols=set(config.market_data.symbols),
        allowed_intervals=set(SUPPORTED_CANDLE_INTERVALS),
    ).read(
        interval=arguments.interval,
        symbols=[arguments.symbol],
        start_date=arguments.start_date,
        end_date=arguments.end_date,
    )

    candles = unified_result.candles

    validate_minimum_history(
        candles=candles,
        required_candles=(maximum_slow_window + 1),
    )

    selected_interval_minutes = interval_minutes(arguments.interval)

    coverage = CandleCoverageAnalyzer(
        interval_minutes=(selected_interval_minutes),
        session_gap_threshold_minutes=(selected_interval_minutes * 5),
    ).analyze(candles)

    print(f"Interval: {arguments.interval} ({selected_interval_minutes} minutes)")

    print(f"Historical source candles: {unified_result.historical_candle_count}")
    print(f"Ticker-derived source candles: {unified_result.ticker_candle_count}")
    print(f"Ticker-over-historical overlaps: {unified_result.overlap_count}")

    print(f"Candles: {coverage.total_candles}")
    print(f"Missing intervals: {coverage.total_missing_intervals}")
    print(f"Session gaps: {coverage.total_session_gaps}")
    print(f"Duplicate intervals: {coverage.total_duplicates}")

    if coverage.total_duplicates > 0:
        raise ValueError("Parameter sweep cannot run with duplicate candle intervals")

    if coverage.total_session_gaps > 0 and not arguments.allow_session_gaps:
        raise ValueError(
            "The candle dataset contains session gaps. "
            "Use --allow-session-gaps only when the "
            "discontinuities are intentional."
        )

    baseline = calculate_buy_and_hold(
        candles=candles,
        initial_cash=arguments.initial_cash,
        fee_rate=arguments.fee_rate,
        slippage_basis_points=(arguments.slippage_bps),
    )

    sweep = MovingAverageParameterSweep(
        initial_cash=arguments.initial_cash,
        allocation_fraction=(arguments.allocation),
        fee_rate=arguments.fee_rate,
        slippage_basis_points=(arguments.slippage_bps),
        liquidate_at_end=True,
    )

    results = sweep.run(
        candles=candles,
        configurations=configurations,
    )

    export_sweep_results(
        results=results,
        path=arguments.output,
    )

    print(f"\nBuy-and-hold return: {baseline.return_percentage}%")
    print(f"Buy-and-hold final equity: {baseline.final_equity}")

    print("\nTop parameter combinations:")

    for rank, result in enumerate(
        results[: arguments.top],
        start=1,
    ):
        profit_factor = (
            str(result.profit_factor) if result.profit_factor is not None else "infinite"
        )

        print(
            f"  {rank}. "
            f"fast={result.fast_window} "
            f"slow={result.slow_window} "
            f"return={result.return_percentage}% "
            f"drawdown="
            f"{result.maximum_drawdown_percentage}% "
            f"trades="
            f"{result.completed_trade_count} "
            f"win_rate="
            f"{result.win_rate_percentage}% "
            f"profit_factor={profit_factor}"
        )

    print(f"\nExported results to: {arguments.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
