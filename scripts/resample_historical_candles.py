import argparse
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import pyarrow.dataset as ds

from trading_bot.market_data.historical_candle import (
    HistoricalCandle,
)
from trading_bot.market_data.historical_resampler import (
    SUPPORTED_TARGET_INTERVALS,
    HistoricalCandleResampler,
)
from trading_bot.storage.historical_candle_parquet import (
    HistoricalCandleParquetWriter,
)


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Date must use YYYY-MM-DD format") from error


def parse_target_intervals(
    value: str,
) -> list[str]:
    intervals = [item.strip() for item in value.split(",") if item.strip()]

    if not intervals:
        raise argparse.ArgumentTypeError("At least one target interval is required")

    unsupported = set(intervals) - set(SUPPORTED_TARGET_INTERVALS)

    if unsupported:
        raise argparse.ArgumentTypeError(
            "Unsupported target intervals: " + ", ".join(sorted(unsupported))
        )

    return intervals


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Resample historical 1m OHLCV candles into higher timeframes")
    )

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
        "--target-intervals",
        type=parse_target_intervals,
        default=["5m", "15m", "1h"],
    )

    parser.add_argument(
        "--historical-directory",
        type=Path,
        default=Path("data/processed/historical_candles"),
    )

    return parser.parse_args()


def read_one_minute_candles(
    directory: Path,
    symbol: str,
    start_date: date,
    end_date: date,
) -> list[HistoricalCandle]:
    if not directory.exists():
        raise FileNotFoundError(f"Historical dataset directory does not exist: {directory}")

    if not any(directory.rglob("*.parquet")):
        raise FileNotFoundError(f"No Parquet files found under: {directory}")

    dataset = ds.dataset(  # type: ignore[no-untyped-call]
        directory,
        format="parquet",
        partitioning="hive",
    )

    expression = (
        (
            ds.field("interval")  # type: ignore[attr-defined,no-untyped-call]
            == "1m"
        )
        & (
            ds.field("symbol")  # type: ignore[attr-defined,no-untyped-call]
            == symbol
        )
        & (
            ds.field("date")  # type: ignore[attr-defined,no-untyped-call]
            >= start_date.isoformat()
        )
        & (
            ds.field("date")  # type: ignore[attr-defined,no-untyped-call]
            <= end_date.isoformat()
        )
    )

    table = dataset.to_table(
        columns=[
            "instrument_id",
            "interval_start",
            "interval_end",
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "base_volume",
            "currency_volume",
            "quote_volume",
            "is_complete",
        ],
        filter=expression,
    )

    candles: list[HistoricalCandle] = []

    for row_number, record in enumerate(
        table.to_pylist(),
        start=1,
    ):
        try:
            interval_start = record["interval_start"]
            interval_end = record["interval_end"]

            if not isinstance(
                interval_start,
                datetime,
            ):
                raise TypeError("interval_start must be datetime")

            if not isinstance(
                interval_end,
                datetime,
            ):
                raise TypeError("interval_end must be datetime")

            candles.append(
                HistoricalCandle(
                    instrument_id=record["instrument_id"],
                    interval="1m",
                    interval_start=(interval_start.astimezone(UTC)),
                    interval_end=(interval_end.astimezone(UTC)),
                    open_price=record["open_price"],
                    high_price=record["high_price"],
                    low_price=record["low_price"],
                    close_price=record["close_price"],
                    base_volume=record["base_volume"],
                    currency_volume=record["currency_volume"],
                    quote_volume=record["quote_volume"],
                    is_complete=record["is_complete"],
                )
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(f"Invalid historical candle row {row_number}: {error}") from error

    candles.sort(
        key=lambda candle: (
            candle.interval_start,
            candle.instrument_id,
        )
    )

    expected_start = datetime.combine(
        start_date,
        time.min,
        tzinfo=UTC,
    )

    expected_end_exclusive = datetime.combine(
        end_date + timedelta(days=1),
        time.min,
        tzinfo=UTC,
    )

    return [
        candle
        for candle in candles
        if (expected_start <= candle.interval_start < expected_end_exclusive)
    ]


def main() -> int:
    arguments = parse_arguments()

    if arguments.end_date < arguments.start_date:
        raise ValueError("end-date must not be before start-date")

    source_candles = read_one_minute_candles(
        directory=arguments.historical_directory,
        symbol=arguments.symbol,
        start_date=arguments.start_date,
        end_date=arguments.end_date,
    )

    print(f"Loaded {len(source_candles)} one-minute candles")

    if not source_candles:
        raise ValueError("The selected query returned no 1m candles")

    writer = HistoricalCandleParquetWriter(arguments.historical_directory)

    resampler = HistoricalCandleResampler(source_interval="1m")

    for target_interval in arguments.target_intervals:
        result = resampler.resample(
            candles=source_candles,
            target_interval=target_interval,
        )

        report = result.report

        print(f"\nResampling 1m -> {target_interval}")
        print(f"  Candidate buckets: {report.candidate_bucket_count}")
        print(f"  Complete candles produced: {report.written_candle_count}")
        print(f"  Partial buckets skipped: {report.skipped_partial_bucket_count}")
        print(f"  Incomplete buckets skipped: {report.skipped_incomplete_bucket_count}")
        print(f"  Duplicate source candles: {report.duplicate_source_candle_count}")

        if result.candles:
            writer.write(result.candles)

            print("  Written to historical dataset")
        else:
            print("  Nothing written")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
