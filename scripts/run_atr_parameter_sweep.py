import argparse
import csv
from datetime import date
from decimal import Decimal, InvalidOperation
from itertools import product
from pathlib import Path

from trading_bot.backtesting.atr_sweep import (
    AtrParameterSweep,
    AtrSweepConfiguration,
    AtrSweepResult,
)
from trading_bot.backtesting.baseline import (
    calculate_buy_and_hold,
)
from trading_bot.backtesting.coverage import (
    CandleCoverageAnalyzer,
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
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Date must use YYYY-MM-DD format") from error


def candle_interval(
    value: str,
) -> str:
    try:
        return parse_candle_interval(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


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
        raise argparse.ArgumentTypeError("All values must be greater than zero")

    return values


def parse_non_negative_integer_list(
    value: str,
) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from error

    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required")

    if any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("All values must be non-negative")

    return values


def parse_decimal_list(
    value: str,
) -> list[Decimal]:
    try:
        values = [Decimal(item.strip()) for item in value.split(",") if item.strip()]
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("Expected comma-separated decimal values") from error

    if not values:
        raise argparse.ArgumentTypeError("At least one decimal value is required")

    if any(not item.is_finite() or item < 0 for item in values):
        raise argparse.ArgumentTypeError("All decimal values must be finite and non-negative")

    return values


def positive_integer(
    value: str,
) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected an integer") from error

    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero")

    return parsed


def positive_decimal(
    value: str,
) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("Expected a decimal value") from error

    if not parsed.is_finite() or parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive finite decimal")

    return parsed


def non_negative_decimal(
    value: str,
) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("Expected a decimal value") from error

    if not parsed.is_finite() or parsed < 0:
        raise argparse.ArgumentTypeError("Value must be a non-negative finite decimal")

    return parsed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run an ATR-filtered moving-average parameter sweep")
    )

    parser.add_argument(
        "--symbol",
        required=True,
        help="Instrument ID, for example BTC-USDT",
    )

    parser.add_argument(
        "--interval",
        type=candle_interval,
        choices=list(SUPPORTED_CANDLE_INTERVALS),
        default="1h",
        help=("Candle interval. Supported values: 1m, 5m, 15m, 1h"),
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
        "--fast-windows",
        type=parse_integer_list,
        required=True,
    )

    parser.add_argument(
        "--slow-windows",
        type=parse_integer_list,
        required=True,
    )

    parser.add_argument(
        "--atr-periods",
        type=parse_integer_list,
        default=[14],
    )

    parser.add_argument(
        "--atr-multipliers",
        type=parse_decimal_list,
        default=[
            Decimal("0.05"),
            Decimal("0.1"),
            Decimal("0.2"),
            Decimal("0.3"),
        ],
    )

    parser.add_argument(
        "--minimum-holding-candles",
        type=parse_non_negative_integer_list,
        default=[0],
    )

    parser.add_argument(
        "--cooldown-candles",
        type=parse_non_negative_integer_list,
        default=[0],
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
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--top",
        type=positive_integer,
        default=20,
    )

    return parser.parse_args()


def create_configurations(
    fast_windows: list[int],
    slow_windows: list[int],
    atr_periods: list[int],
    atr_multipliers: list[Decimal],
    minimum_holding_candles: list[int],
    cooldown_candles: list[int],
) -> list[AtrSweepConfiguration]:
    return [
        AtrSweepConfiguration(
            fast_window=fast_window,
            slow_window=slow_window,
            atr_period=atr_period,
            atr_multiplier=atr_multiplier,
            minimum_holding_candles=(minimum_holding),
            cooldown_candles=cooldown,
        )
        for (
            fast_window,
            slow_window,
            atr_period,
            atr_multiplier,
            minimum_holding,
            cooldown,
        ) in product(
            fast_windows,
            slow_windows,
            atr_periods,
            atr_multipliers,
            minimum_holding_candles,
            cooldown_candles,
        )
        if fast_window < slow_window
    ]


def export_results(
    path: Path,
    results: list[AtrSweepResult],
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
                "fast_window",
                "slow_window",
                "atr_period",
                "atr_multiplier",
                "minimum_holding_candles",
                "cooldown_candles",
                "return_percentage",
                "maximum_drawdown_percentage",
                "completed_trade_count",
                "win_rate_percentage",
                "profit_factor",
                "total_fees",
            ]
        )

        for result in results:
            writer.writerow(
                [
                    result.fast_window,
                    result.slow_window,
                    result.atr_period,
                    str(result.atr_multiplier),
                    result.minimum_holding_candles,
                    result.cooldown_candles,
                    str(result.return_percentage),
                    str(result.maximum_drawdown_percentage),
                    result.completed_trade_count,
                    str(result.win_rate_percentage),
                    (str(result.profit_factor) if result.profit_factor is not None else "infinite"),
                    str(result.total_fees),
                ]
            )


def main() -> int:
    arguments = parse_arguments()

    if arguments.end_date < arguments.start_date:
        raise ValueError("end-date must not be before start-date")

    if arguments.allocation > Decimal("1"):
        raise ValueError("allocation must not exceed 1")

    if arguments.fee_rate >= Decimal("1"):
        raise ValueError("fee-rate must be below 1")

    if arguments.slippage_bps >= Decimal("10000"):
        raise ValueError("slippage-bps must be below 10000")

    configurations = create_configurations(
        fast_windows=arguments.fast_windows,
        slow_windows=arguments.slow_windows,
        atr_periods=arguments.atr_periods,
        atr_multipliers=(arguments.atr_multipliers),
        minimum_holding_candles=(arguments.minimum_holding_candles),
        cooldown_candles=(arguments.cooldown_candles),
    )

    if not configurations:
        raise ValueError(
            "No valid parameter combinations were "
            "created. Every configuration requires "
            "fast_window < slow_window."
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

    if not candles:
        raise ValueError("The selected dataset query returned no candles")

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
        raise ValueError("Parameter sweep cannot run with duplicate candle intervals")

    if coverage.total_missing_intervals > 0:
        raise ValueError("Parameter sweep cannot run with missing candle intervals")

    if coverage.total_session_gaps > 0:
        raise ValueError("Parameter sweep cannot run with session gaps")

    maximum_required_history = max(
        max(
            configuration.slow_window,
            configuration.atr_period,
        )
        for configuration in configurations
    )

    if len(candles) <= maximum_required_history:
        raise ValueError(
            "Insufficient candle history. "
            f"Required more than "
            f"{maximum_required_history} candles, "
            f"available={len(candles)}"
        )

    buy_and_hold = calculate_buy_and_hold(
        candles=candles,
        initial_cash=arguments.initial_cash,
        fee_rate=arguments.fee_rate,
        slippage_basis_points=(arguments.slippage_bps),
    )

    print(f"\nBuy-and-hold return: {buy_and_hold.return_percentage}%")

    print(f"Buy-and-hold final equity: {buy_and_hold.final_equity}")

    estimated_round_trip_cost_rate = arguments.fee_rate * Decimal(
        "2"
    ) + arguments.slippage_bps / Decimal("10000") * Decimal("2")

    print(f"\nEstimated round-trip cost rate: {estimated_round_trip_cost_rate}")

    print(
        f"Estimated round-trip cost percentage: {estimated_round_trip_cost_rate * Decimal('100')}%"
    )

    print(f"Parameter combinations: {len(configurations)}")

    sweep = AtrParameterSweep(
        initial_cash=arguments.initial_cash,
        allocation_fraction=(arguments.allocation),
        fee_rate=arguments.fee_rate,
        slippage_basis_points=(arguments.slippage_bps),
        estimated_round_trip_cost_rate=(estimated_round_trip_cost_rate),
        liquidate_at_end=True,
    )

    results = sweep.run(
        candles=candles,
        configurations=configurations,
    )

    print("\nTop ATR-filtered parameter combinations:")

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
            f"atr_period={result.atr_period} "
            f"atr_multiplier="
            f"{result.atr_multiplier} "
            f"hold="
            f"{result.minimum_holding_candles} "
            f"cooldown="
            f"{result.cooldown_candles} "
            f"return="
            f"{result.return_percentage}% "
            f"drawdown="
            f"{result.maximum_drawdown_percentage}% "
            f"trades="
            f"{result.completed_trade_count} "
            f"win_rate="
            f"{result.win_rate_percentage}% "
            f"profit_factor="
            f"{profit_factor} "
            f"fees={result.total_fees}"
        )

    export_results(
        path=arguments.output,
        results=results,
    )

    print(f"\nExported results to: {arguments.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
