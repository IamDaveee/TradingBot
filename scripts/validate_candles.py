import argparse
from datetime import date

from trading_bot.settings import load_settings
from trading_bot.storage.candle_parquet_reader import (
    CandleDatasetReader,
)
from trading_bot.validation.candle_validator import (
    CandleValidator,
)


def parse_date(
    value: str,
) -> date:
    return date.fromisoformat(value)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Validate processed candle data"))

    parser.add_argument(
        "--interval",
        default="1m",
    )

    parser.add_argument(
        "--symbol",
        action="append",
        dest="symbols",
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
        "--max-issues",
        type=int,
        default=20,
    )

    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()

    if arguments.max_issues <= 0:
        raise ValueError("max-issues must be greater than zero")

    _, config = load_settings()

    reader = CandleDatasetReader(
        dataset_directory=(config.storage.candle_directory),
        allowed_symbols=set(config.market_data.symbols),
        allowed_intervals={"1m"},
    )

    table = reader.read(
        interval=arguments.interval,
        symbols=arguments.symbols,
        start_date=arguments.start_date,
        end_date=arguments.end_date,
    )

    validator = CandleValidator()

    report = validator.validate(
        table=table,
        interval_minutes=1,
    )

    print(f"Total records: {report.total_records}")
    print(f"Invalid records: {report.invalid_records}")
    print(f"Duplicate intervals: {report.duplicate_intervals}")
    print(f"Missing intervals: {report.missing_intervals}")
    print(f"Session gaps: {report.session_gaps}")
    print(f"Valid: {report.is_valid}")

    if report.issues:
        print("\nIssues:")

        for issue in report.issues[: arguments.max_issues]:
            print(
                f"  [{issue.code}] "
                f"symbol={issue.symbol} "
                f"interval={issue.interval_start} "
                f"{issue.message}"
            )

    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
