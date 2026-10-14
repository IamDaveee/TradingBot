import argparse
from datetime import date

from trading_bot.settings import load_settings
from trading_bot.storage.parquet_reader import (
    TickerDatasetReader,
)


def parse_date(
    value: str,
) -> date:
    return date.fromisoformat(value)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Inspect the processed ticker dataset"))

    parser.add_argument(
        "--symbol",
        action="append",
        dest="symbols",
        help=("Symbol to load. May be supplied multiple times."),
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
        "--limit",
        type=int,
        default=10,
    )

    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    _, config = load_settings()

    if arguments.limit <= 0:
        raise ValueError("Limit must be greater than zero")

    reader = TickerDatasetReader(
        dataset_directory=(config.storage.processed_directory),
        allowed_symbols=set(config.market_data.symbols),
    )

    print(
        "Available symbols:",
        reader.available_symbols(),
    )

    for symbol in reader.available_symbols():
        print(
            f"Available dates for {symbol}:",
            reader.available_dates(symbol),
        )

    table = reader.read(
        symbols=arguments.symbols,
        start_date=arguments.start_date,
        end_date=arguments.end_date,
    )

    print(f"\nRows returned: {table.num_rows}")
    print(f"Columns: {table.column_names}")
    print("\nPreview:")
    print(table.slice(0, arguments.limit))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
