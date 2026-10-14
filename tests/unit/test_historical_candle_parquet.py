from datetime import (
    UTC,
    datetime,
    timedelta,
)
from decimal import Decimal
from pathlib import Path

import pyarrow.parquet as pq

from trading_bot.market_data.historical_candle import (
    HistoricalCandle,
)
from trading_bot.storage.historical_candle_parquet import (
    HistoricalCandleParquetWriter,
)


def candle(
    minute: int,
    close_price: str = "105",
    symbol: str = "BTC-USDT",
) -> HistoricalCandle:
    start = datetime(
        2026,
        6,
        1,
        12,
        minute,
        tzinfo=UTC,
    )

    return HistoricalCandle(
        instrument_id=symbol,
        interval="1m",
        interval_start=start,
        interval_end=(start + timedelta(minutes=1)),
        open_price=Decimal("100"),
        high_price=Decimal("110"),
        low_price=Decimal("90"),
        close_price=Decimal(close_price),
        base_volume=Decimal("12.5"),
        currency_volume=Decimal("12.5"),
        quote_volume=Decimal("1300"),
        is_complete=True,
    )


def test_writes_hive_partition(
    tmp_path: Path,
) -> None:
    writer = HistoricalCandleParquetWriter(dataset_directory=tmp_path)

    result = writer.write(
        [
            candle(0),
            candle(1),
        ]
    )

    expected_path = (
        tmp_path / "interval=1m" / "symbol=BTC-USDT" / "date=2026-06-01" / "part-history.parquet"
    )

    assert expected_path.exists()
    assert result.partition_count == 1
    assert result.written_candles == 2

    table = pq.ParquetFile(  # type: ignore[no-untyped-call]
        expected_path
    ).read()

    assert table.num_rows == 2


def test_repeated_write_deduplicates(
    tmp_path: Path,
) -> None:
    writer = HistoricalCandleParquetWriter(dataset_directory=tmp_path)

    writer.write(
        [
            candle(0),
            candle(1),
        ]
    )

    result = writer.write(
        [
            candle(1),
            candle(2),
        ]
    )

    expected_path = (
        tmp_path / "interval=1m" / "symbol=BTC-USDT" / "date=2026-06-01" / "part-history.parquet"
    )

    table = pq.ParquetFile(  # type: ignore[no-untyped-call]
        expected_path
    ).read()

    assert table.num_rows == 3
    assert result.existing_candles == 2
    assert result.duplicate_candles == 1
    assert result.written_candles == 3


def test_new_download_replaces_existing_value(
    tmp_path: Path,
) -> None:
    writer = HistoricalCandleParquetWriter(dataset_directory=tmp_path)

    writer.write(
        [
            candle(
                0,
                close_price="101",
            )
        ]
    )

    writer.write(
        [
            candle(
                0,
                close_price="109",
            )
        ]
    )

    path = tmp_path / "interval=1m" / "symbol=BTC-USDT" / "date=2026-06-01" / "part-history.parquet"

    table = pq.ParquetFile(  # type: ignore[no-untyped-call]
        path
    ).read()

    records = table.to_pylist()

    assert len(records) == 1

    assert records[0]["close_price"] == Decimal("109.000000000000000000")


def test_excludes_local_overlap(
    tmp_path: Path,
) -> None:
    writer = HistoricalCandleParquetWriter(dataset_directory=tmp_path)

    first = candle(0)
    second = candle(1)

    result = writer.write(
        candles=[
            first,
            second,
        ],
        excluded_timestamps={
            (
                first.instrument_id,
                first.interval_start,
            )
        },
    )

    assert result.excluded_candles == 1
    assert result.written_candles == 1


def test_partitions_multiple_dates(
    tmp_path: Path,
) -> None:
    first = candle(0)

    second_start = datetime(
        2026,
        6,
        2,
        0,
        0,
        tzinfo=UTC,
    )

    second = HistoricalCandle(
        instrument_id="BTC-USDT",
        interval="1m",
        interval_start=second_start,
        interval_end=(second_start + timedelta(minutes=1)),
        open_price=Decimal("100"),
        high_price=Decimal("110"),
        low_price=Decimal("90"),
        close_price=Decimal("105"),
        base_volume=Decimal("1"),
        currency_volume=Decimal("1"),
        quote_volume=Decimal("100"),
        is_complete=True,
    )

    result = HistoricalCandleParquetWriter(dataset_directory=tmp_path).write(
        [
            first,
            second,
        ]
    )

    assert result.partition_count == 2
    assert len(result.written_files) == 2
