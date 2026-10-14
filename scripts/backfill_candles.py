import argparse
import asyncio
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds

from trading_bot.exchanges.okx.historical_candle_client import (
    OkxHistoricalCandleClient,
)
from trading_bot.settings import load_settings
from trading_bot.storage.historical_candle_parquet import (
    HistoricalCandleParquetWriter,
)


def parse_date(
    value: str,
) -> date:
    return date.fromisoformat(value)


def positive_integer(
    value: str,
) -> int:
    parsed = int(value)

    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero")

    return parsed


def non_negative_decimal(
    value: str,
) -> Decimal:
    parsed = Decimal(value)

    if not parsed.is_finite() or parsed < 0:
        raise argparse.ArgumentTypeError("Value must be a non-negative decimal")

    return parsed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Backfill historical OKX candles"))

    parser.add_argument(
        "--symbol",
        required=True,
    )

    parser.add_argument(
        "--interval",
        default="1m",
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
        "--output-directory",
        type=Path,
        default=Path("data/processed/historical_candles"),
    )

    parser.add_argument(
        "--request-delay",
        type=non_negative_decimal,
        default=Decimal("0.12"),
    )

    parser.add_argument(
        "--maximum-attempts",
        type=positive_integer,
        default=5,
    )

    parser.add_argument(
        "--include-incomplete",
        action="store_true",
    )

    parser.add_argument(
        "--keep-overlapping-local-candles",
        action="store_true",
    )

    return parser.parse_args()


def utc_day_start(
    value: date,
) -> datetime:
    return datetime.combine(
        value,
        time.min,
        tzinfo=UTC,
    )


def load_local_candle_timestamps(
    dataset_directory: Path,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
) -> set[tuple[str, datetime]]:
    if not dataset_directory.exists():
        return set()

    dataset = ds.dataset(  # type: ignore[no-untyped-call]
        dataset_directory,
        format="parquet",
        partitioning="hive",
    )

    if "interval_start" not in (dataset.schema.names):
        return set()

    expression = (
        (
            ds.field("interval")  # type: ignore[attr-defined,no-untyped-call]
            == interval
        )
        & (
            ds.field("symbol")  # type: ignore[attr-defined,no-untyped-call]
            == symbol
        )
        & (
            ds.field("interval_start")  # type: ignore[attr-defined,no-untyped-call]
            >= pa.scalar(
                start,
                type=pa.timestamp(
                    "us",
                    tz="UTC",
                ),
            )
        )
        & (
            ds.field("interval_start")  # type: ignore[attr-defined,no-untyped-call]
            < pa.scalar(
                end,
                type=pa.timestamp(
                    "us",
                    tz="UTC",
                ),
            )
        )
    )

    table = dataset.to_table(
        columns=[
            "instrument_id",
            "interval_start",
        ],
        filter=expression,
    )

    timestamps: set[tuple[str, datetime]] = set()

    for record in table.to_pylist():
        instrument_id = record.get("instrument_id")

        interval_start = record.get("interval_start")

        if isinstance(instrument_id, str) and isinstance(
            interval_start,
            datetime,
        ):
            timestamps.add(
                (
                    instrument_id,
                    interval_start,
                )
            )

    return timestamps


def calculate_missing_intervals(
    timestamps: list[datetime],
    interval_duration: timedelta,
) -> int:
    unique_timestamps = sorted(set(timestamps))

    missing = 0

    for previous, current in zip(
        unique_timestamps,
        unique_timestamps[1:],
        strict=False,
    ):
        difference = current - previous

        if difference <= interval_duration:
            continue

        missing += int(difference / interval_duration) - 1

    return missing


def interval_duration(
    interval: str,
) -> timedelta:
    durations = {
        "1m": timedelta(minutes=1),
        "3m": timedelta(minutes=3),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1H": timedelta(hours=1),
        "2H": timedelta(hours=2),
        "4H": timedelta(hours=4),
        "6H": timedelta(hours=6),
        "12H": timedelta(hours=12),
        "1D": timedelta(days=1),
    }

    try:
        return durations[interval]
    except KeyError as error:
        raise ValueError(f"Unsupported interval: {interval}") from error


async def run_backfill(
    arguments: argparse.Namespace,
) -> int:
    if arguments.end_date < arguments.start_date:
        raise ValueError("end-date must not be before start-date")

    start = utc_day_start(arguments.start_date)

    # CLI end-date is inclusive.
    end = utc_day_start(arguments.end_date + timedelta(days=1))

    _, config = load_settings()

    if arguments.symbol not in config.market_data.symbols:
        raise ValueError(f"Symbol is not enabled in settings: {arguments.symbol}")

    print(f"Downloading {arguments.symbol} {arguments.interval} candles")

    print(f"Range: {start.isoformat()} to {end.isoformat()} (end exclusive)")

    async with OkxHistoricalCandleClient(
        maximum_attempts=(arguments.maximum_attempts),
        request_delay_seconds=float(arguments.request_delay),
    ) as client:
        candles = await client.fetch_range(
            instrument_id=arguments.symbol,
            interval=arguments.interval,
            start=start,
            end=end,
            include_incomplete=(arguments.include_incomplete),
        )

    if not candles:
        print("No historical candles returned.")
        return 1

    local_timestamps: set[tuple[str, datetime]] = set()

    if not (arguments.keep_overlapping_local_candles):
        local_timestamps = load_local_candle_timestamps(
            dataset_directory=(config.storage.candle_directory),
            symbol=arguments.symbol,
            interval=arguments.interval,
            start=start,
            end=end,
        )

    writer = HistoricalCandleParquetWriter(dataset_directory=(arguments.output_directory))

    result = writer.write(
        candles=candles,
        excluded_timestamps=(local_timestamps),
    )

    duration = interval_duration(arguments.interval)

    downloaded_missing = calculate_missing_intervals(
        timestamps=[candle.interval_start for candle in candles],
        interval_duration=duration,
    )

    complete_count = sum(1 for candle in candles if candle.is_complete)

    incomplete_count = len(candles) - complete_count

    print("\nBackfill summary:")
    print(f"  Downloaded candles: {len(candles)}")
    print(f"  Complete candles: {complete_count}")
    print(f"  Incomplete candles: {incomplete_count}")
    print(f"  Existing historical candles: {result.existing_candles}")
    print(f"  Duplicate candles: {result.duplicate_candles}")
    print(f"  Overlapping local candles skipped: {result.excluded_candles}")
    print(f"  Candles in written partitions: {result.written_candles}")
    print(f"  Written partitions: {result.partition_count}")
    print(f"  Missing intervals in download: {downloaded_missing}")

    if result.written_files:
        print("\nWritten files:")

        for path in result.written_files:
            print(f"  {path}")

    return 0


def main() -> int:
    arguments = parse_arguments()

    return asyncio.run(run_backfill(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
