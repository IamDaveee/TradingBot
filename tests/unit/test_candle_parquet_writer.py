from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from trading_bot.market_data.candle import (
    TickerCandle,
)
from trading_bot.storage.candle_parquet_writer import (
    CandleParquetWriteError,
    CandleParquetWriter,
)


def create_candle(
    symbol: str = "BTC-USDT",
    minute: int = 0,
) -> TickerCandle:
    interval_start = datetime(
        2026,
        6,
        27,
        12,
        minute,
        tzinfo=UTC,
    )

    interval_end = interval_start + timedelta(minutes=1)

    return TickerCandle(
        instrument_id=symbol,
        interval_start=interval_start,
        interval_end=interval_end,
        open_price=Decimal("100"),
        high_price=Decimal("103"),
        low_price=Decimal("98"),
        close_price=Decimal("101"),
        first_bid_price=Decimal("99.9"),
        last_bid_price=Decimal("100.9"),
        first_ask_price=Decimal("100.1"),
        last_ask_price=Decimal("101.1"),
        minimum_spread=Decimal("0.1"),
        maximum_spread=Decimal("0.4"),
        average_spread=Decimal("0.2"),
        observed_size=Decimal("10"),
        tick_count=4,
        first_exchange_timestamp=(interval_start),
        last_exchange_timestamp=(interval_start + timedelta(seconds=50)),
        first_received_timestamp=(interval_start),
        last_received_timestamp=(interval_start + timedelta(seconds=50)),
    )


def test_writer_creates_partitioned_file(
    tmp_path: Path,
) -> None:
    writer = CandleParquetWriter(output_directory=tmp_path)

    written_files = writer.write(
        candles=[
            create_candle(),
            create_candle(symbol="ETH-USDT"),
        ],
        interval_minutes=1,
    )

    assert len(written_files) == 2

    btc_file = (
        tmp_path
        / "interval=1m"
        / "symbol=BTC-USDT"
        / "date=2026-06-27"
        / "part-candles-2026-06-27.parquet"
    )

    assert btc_file.exists()

    table = pq.read_table(btc_file)

    assert table.num_rows == 1

    assert table.column("instrument_id")[0].as_py() == "BTC-USDT"

    assert table.column("tick_count")[0].as_py() == 4


def test_writer_writes_multiple_candles(
    tmp_path: Path,
) -> None:
    writer = CandleParquetWriter(output_directory=tmp_path)

    written_files = writer.write(
        candles=[
            create_candle(minute=0),
            create_candle(minute=1),
        ],
        interval_minutes=1,
    )

    assert len(written_files) == 1

    table = pq.read_table(written_files[0])

    assert table.num_rows == 2


def test_writer_rejects_empty_input(
    tmp_path: Path,
) -> None:
    writer = CandleParquetWriter(output_directory=tmp_path)

    with pytest.raises(CandleParquetWriteError):
        writer.write(
            candles=[],
            interval_minutes=1,
        )


def test_writer_rejects_invalid_interval(
    tmp_path: Path,
) -> None:
    writer = CandleParquetWriter(output_directory=tmp_path)

    with pytest.raises(ValueError):
        writer.write(
            candles=[create_candle()],
            interval_minutes=0,
        )


def test_writer_refuses_overwrite(
    tmp_path: Path,
) -> None:
    writer = CandleParquetWriter(output_directory=tmp_path)

    writer.write(
        candles=[create_candle()],
        interval_minutes=1,
    )

    with pytest.raises(FileExistsError):
        writer.write(
            candles=[create_candle()],
            interval_minutes=1,
        )


def test_writer_can_overwrite(
    tmp_path: Path,
) -> None:
    writer = CandleParquetWriter(output_directory=tmp_path)

    writer.write(
        candles=[create_candle()],
        interval_minutes=1,
    )

    written_files = writer.write(
        candles=[create_candle()],
        interval_minutes=1,
        overwrite=True,
    )

    assert len(written_files) == 1


def test_writer_quantizes_repeating_decimal(
    tmp_path: Path,
) -> None:
    candle = create_candle()

    candle = TickerCandle(
        instrument_id=candle.instrument_id,
        interval_start=candle.interval_start,
        interval_end=candle.interval_end,
        open_price=candle.open_price,
        high_price=candle.high_price,
        low_price=candle.low_price,
        close_price=candle.close_price,
        first_bid_price=candle.first_bid_price,
        last_bid_price=candle.last_bid_price,
        first_ask_price=candle.first_ask_price,
        last_ask_price=candle.last_ask_price,
        minimum_spread=Decimal("0.01"),
        maximum_spread=Decimal("0.02"),
        average_spread=(Decimal("0.04") / Decimal("3")),
        observed_size=candle.observed_size,
        tick_count=candle.tick_count,
        first_exchange_timestamp=(candle.first_exchange_timestamp),
        last_exchange_timestamp=(candle.last_exchange_timestamp),
        first_received_timestamp=(candle.first_received_timestamp),
        last_received_timestamp=(candle.last_received_timestamp),
    )

    writer = CandleParquetWriter(output_directory=tmp_path)

    written_files = writer.write(
        candles=[candle],
        interval_minutes=1,
    )

    table = pq.read_table(written_files[0])

    assert table.column("average_spread")[0].as_py() == Decimal("0.013333333333333333")
