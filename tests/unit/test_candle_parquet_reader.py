from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from trading_bot.storage.candle_parquet_reader import (
    CandleDatasetReader,
)

SCHEMA = pa.schema(
    [
        pa.field(
            "instrument_id",
            pa.string(),
            nullable=False,
        ),
        pa.field(
            "interval_start",
            pa.timestamp("us", tz="UTC"),
            nullable=False,
        ),
        pa.field(
            "interval_end",
            pa.timestamp("us", tz="UTC"),
            nullable=False,
        ),
        pa.field(
            "open_price",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "high_price",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "low_price",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "close_price",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "first_bid_price",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "last_bid_price",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "first_ask_price",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "last_ask_price",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "minimum_spread",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "maximum_spread",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "average_spread",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "observed_size",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "tick_count",
            pa.int64(),
            nullable=False,
        ),
        pa.field(
            "first_exchange_timestamp",
            pa.timestamp("us", tz="UTC"),
            nullable=False,
        ),
        pa.field(
            "last_exchange_timestamp",
            pa.timestamp("us", tz="UTC"),
            nullable=False,
        ),
        pa.field(
            "first_received_timestamp",
            pa.timestamp("us", tz="UTC"),
            nullable=False,
        ),
        pa.field(
            "last_received_timestamp",
            pa.timestamp("us", tz="UTC"),
            nullable=False,
        ),
    ]
)


def write_partition(
    root: Path,
    symbol: str,
    partition_date: str,
) -> None:
    start = datetime.fromisoformat(f"{partition_date}T12:00:00+00:00").astimezone(UTC)

    directory = root / "interval=1m" / f"symbol={symbol}" / f"date={partition_date}"

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    table = pa.Table.from_pylist(
        [
            {
                "instrument_id": symbol,
                "interval_start": start,
                "interval_end": start.replace(minute=1),
                "open_price": Decimal("100"),
                "high_price": Decimal("101"),
                "low_price": Decimal("99"),
                "close_price": Decimal("100.5"),
                "first_bid_price": Decimal("99.9"),
                "last_bid_price": Decimal("100.4"),
                "first_ask_price": Decimal("100.1"),
                "last_ask_price": Decimal("100.6"),
                "minimum_spread": Decimal("0.1"),
                "maximum_spread": Decimal("0.3"),
                "average_spread": Decimal("0.2"),
                "observed_size": Decimal("10"),
                "tick_count": 5,
                "first_exchange_timestamp": start,
                "last_exchange_timestamp": start,
                "first_received_timestamp": start,
                "last_received_timestamp": start,
            }
        ],
        schema=SCHEMA,
    )

    pq.write_table(  # type: ignore[no-untyped-call]
        table,
        directory / "part.parquet",
    )


def test_reader_filters_symbol(
    tmp_path: Path,
) -> None:
    write_partition(
        tmp_path,
        "BTC-USDT",
        "2026-06-27",
    )
    write_partition(
        tmp_path,
        "ETH-USDT",
        "2026-06-27",
    )

    reader = CandleDatasetReader(
        dataset_directory=tmp_path,
        allowed_symbols={
            "BTC-USDT",
            "ETH-USDT",
        },
    )

    table = reader.read(symbols=["BTC-USDT"])

    assert table.num_rows == 1
    assert table.column("instrument_id")[0].as_py() == "BTC-USDT"


def test_reader_reports_available_symbols(
    tmp_path: Path,
) -> None:
    write_partition(
        tmp_path,
        "BTC-USDT",
        "2026-06-27",
    )
    write_partition(
        tmp_path,
        "ETH-USDT",
        "2026-06-27",
    )

    reader = CandleDatasetReader(
        dataset_directory=tmp_path,
        allowed_symbols={
            "BTC-USDT",
            "ETH-USDT",
        },
    )

    assert reader.available_symbols() == [
        "BTC-USDT",
        "ETH-USDT",
    ]


def test_reader_reports_available_dates(
    tmp_path: Path,
) -> None:
    write_partition(
        tmp_path,
        "BTC-USDT",
        "2026-06-27",
    )

    reader = CandleDatasetReader(
        dataset_directory=tmp_path,
        allowed_symbols={"BTC-USDT"},
    )

    assert reader.available_dates(symbol="BTC-USDT") == [date(2026, 6, 27)]


def test_reader_rejects_unknown_symbol(
    tmp_path: Path,
) -> None:
    write_partition(
        tmp_path,
        "BTC-USDT",
        "2026-06-27",
    )

    reader = CandleDatasetReader(
        dataset_directory=tmp_path,
        allowed_symbols={"BTC-USDT"},
    )

    with pytest.raises(ValueError):
        reader.read(symbols=["DOGE-USDT"])
