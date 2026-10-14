from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from trading_bot.storage.parquet_reader import (
    TickerDatasetError,
    TickerDatasetReader,
)

SCHEMA = pa.schema(
    [
        pa.field(
            "instrument_id",
            pa.string(),
            nullable=False,
        ),
        pa.field(
            "last_price",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "last_size",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "best_ask_price",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "best_ask_size",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "best_bid_price",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "best_bid_size",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "spread",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "spread_percentage",
            pa.float64(),
            nullable=False,
        ),
        pa.field(
            "exchange_timestamp",
            pa.timestamp(
                "us",
                tz="UTC",
            ),
            nullable=False,
        ),
        pa.field(
            "received_timestamp",
            pa.timestamp(
                "us",
                tz="UTC",
            ),
            nullable=False,
        ),
        pa.field(
            "written_timestamp",
            pa.timestamp(
                "us",
                tz="UTC",
            ),
            nullable=False,
        ),
    ]
)


def write_partition(
    root: Path,
    symbol: str,
    partition_date: str,
) -> None:
    directory = root / f"symbol={symbol}" / f"date={partition_date}"

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    exchange_timestamp = datetime.fromisoformat(f"{partition_date}T12:00:00+00:00").astimezone(UTC)

    received_timestamp = datetime.fromisoformat(
        f"{partition_date}T12:00:00.100000+00:00"
    ).astimezone(UTC)

    written_timestamp = datetime.fromisoformat(
        f"{partition_date}T12:00:00.200000+00:00"
    ).astimezone(UTC)

    table = pa.Table.from_pylist(
        [
            {
                "instrument_id": symbol,
                "last_price": Decimal("100"),
                "last_size": Decimal("1"),
                "best_ask_price": Decimal("100.1"),
                "best_ask_size": Decimal("2"),
                "best_bid_price": Decimal("99.9"),
                "best_bid_size": Decimal("3"),
                "spread": Decimal("0.2"),
                "spread_percentage": 0.2,
                "exchange_timestamp": (exchange_timestamp),
                "received_timestamp": (received_timestamp),
                "written_timestamp": (written_timestamp),
            }
        ],
        schema=SCHEMA,
    )

    pq.write_table(  # type: ignore[no-untyped-call]
        table,
        directory / "part.parquet",
    )


def create_reader(
    root: Path,
) -> TickerDatasetReader:
    return TickerDatasetReader(
        dataset_directory=root,
        allowed_symbols={
            "BTC-USDT",
            "ETH-USDT",
        },
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

    reader = create_reader(tmp_path)

    table = reader.read(symbols=["BTC-USDT"])

    assert table.num_rows == 1
    assert table.column("instrument_id")[0].as_py() == "BTC-USDT"


def test_reader_filters_date_range(
    tmp_path: Path,
) -> None:
    write_partition(
        tmp_path,
        "BTC-USDT",
        "2026-06-26",
    )
    write_partition(
        tmp_path,
        "BTC-USDT",
        "2026-06-27",
    )

    reader = create_reader(tmp_path)

    table = reader.read(
        symbols=["BTC-USDT"],
        start_date=date(2026, 6, 27),
        end_date=date(2026, 6, 27),
    )

    assert table.num_rows == 1


def test_reader_rejects_unknown_symbol(
    tmp_path: Path,
) -> None:
    write_partition(
        tmp_path,
        "BTC-USDT",
        "2026-06-27",
    )

    reader = create_reader(tmp_path)

    with pytest.raises(ValueError):
        reader.read(symbols=["DOGE-USDT"])


def test_reader_rejects_invalid_date_range(
    tmp_path: Path,
) -> None:
    write_partition(
        tmp_path,
        "BTC-USDT",
        "2026-06-27",
    )

    reader = create_reader(tmp_path)

    with pytest.raises(ValueError):
        reader.read(
            start_date=date(2026, 6, 28),
            end_date=date(2026, 6, 27),
        )


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

    reader = create_reader(tmp_path)

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
        "2026-06-26",
    )
    write_partition(
        tmp_path,
        "BTC-USDT",
        "2026-06-27",
    )

    reader = create_reader(tmp_path)

    assert reader.available_dates("BTC-USDT") == [
        date(2026, 6, 26),
        date(2026, 6, 27),
    ]


def test_reader_rejects_empty_dataset(
    tmp_path: Path,
) -> None:
    reader = create_reader(tmp_path)

    with pytest.raises(TickerDatasetError):
        reader.read()
