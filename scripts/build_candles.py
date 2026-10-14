import argparse
from datetime import date

from trading_bot.market_data.candle_aggregator import (
    TickerCandleAggregator,
)
from trading_bot.settings import load_settings
from trading_bot.storage.candle_parquet_writer import (
    CandleParquetWriter,
)
from trading_bot.storage.parquet_reader import (
    TickerDatasetReader,
)


def parse_date(
    value: str,
) -> date:
    return date.fromisoformat(value)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Build candle Parquet data from processed ticker data")
    )

    parser.add_argument(
        "--symbol",
        action="append",
        dest="symbols",
        help=("Symbol to process. May be supplied multiple times."),
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
        "--interval-minutes",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()

    if arguments.interval_minutes <= 0:
        raise ValueError("interval-minutes must be greater than zero")

    _, config = load_settings()

    reader = TickerDatasetReader(
        dataset_directory=(config.storage.processed_directory),
        allowed_symbols=set(config.market_data.symbols),
    )

    ticker_table = reader.read(
        symbols=arguments.symbols,
        start_date=arguments.start_date,
        end_date=arguments.end_date,
        columns=[
            "instrument_id",
            "last_price",
            "last_size",
            "best_bid_price",
            "best_ask_price",
            "spread",
            "exchange_timestamp",
            "received_timestamp",
        ],
    )

    aggregator = TickerCandleAggregator()

    candles = aggregator.aggregate(
        table=ticker_table,
        interval_minutes=(arguments.interval_minutes),
    )

    writer = CandleParquetWriter(output_directory=(config.storage.candle_directory))

    written_files = writer.write(
        candles=candles,
        interval_minutes=(arguments.interval_minutes),
        overwrite=arguments.overwrite,
    )

    print(f"Read {ticker_table.num_rows} ticker rows")
    print(f"Created {len(candles)} candles")
    print(f"Wrote {len(written_files)} Parquet file(s):")

    for file_path in written_files:
        print(f"  {file_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
