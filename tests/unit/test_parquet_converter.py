import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from trading_bot.storage.parquet_converter import (
    TickerParquetConversionError,
    TickerParquetConverter,
)


def create_record(
    symbol: str = "BTC-USDT",
) -> dict[str, str]:
    return {
        "instrument_id": symbol,
        "last_price": "60000.1",
        "last_size": "0.01",
        "best_ask_price": "60000.2",
        "best_ask_size": "1.2",
        "best_bid_price": "60000.1",
        "best_bid_size": "1.1",
        "spread": "0.1",
        "spread_percentage": "0.000166",
        "exchange_timestamp": ("2026-06-27T14:20:25.100000+00:00"),
        "received_timestamp": ("2026-06-27T14:20:25.200000+00:00"),
        "written_timestamp": ("2026-06-27T14:20:25.300000+00:00"),
    }


def write_jsonl(
    file_path: Path,
    records: list[dict[str, str]],
) -> None:
    content = "".join(json.dumps(record) + "\n" for record in records)

    file_path.write_text(
        content,
        encoding="utf-8",
    )


def test_converter_creates_symbol_partition(
    tmp_path: Path,
) -> None:
    input_file = tmp_path / ("tickers-2026-06-27.jsonl")
    output_directory = tmp_path / "processed"

    write_jsonl(
        input_file,
        [
            create_record("BTC-USDT"),
            create_record("ETH-USDT"),
        ],
    )

    converter = TickerParquetConverter(
        output_directory=output_directory,
        allowed_symbols={
            "BTC-USDT",
            "ETH-USDT",
        },
    )

    written_files = converter.convert_file(input_file)

    assert len(written_files) == 2

    btc_file = (
        output_directory / "symbol=BTC-USDT" / "date=2026-06-27" / "part-tickers-2026-06-27.parquet"
    )

    assert btc_file.exists()

    table = pq.read_table(btc_file)

    assert table.num_rows == 1
    assert table.column("instrument_id")[0].as_py() == "BTC-USDT"


def test_converter_rejects_unknown_symbol(
    tmp_path: Path,
) -> None:
    input_file = tmp_path / "tickers.jsonl"

    write_jsonl(
        input_file,
        [create_record("DOGE-USDT")],
    )

    converter = TickerParquetConverter(
        output_directory=tmp_path / "processed",
        allowed_symbols={"BTC-USDT"},
    )

    with pytest.raises(TickerParquetConversionError):
        converter.convert_file(input_file)


def test_converter_refuses_to_overwrite(
    tmp_path: Path,
) -> None:
    input_file = tmp_path / ("tickers-2026-06-27.jsonl")

    write_jsonl(
        input_file,
        [create_record()],
    )

    converter = TickerParquetConverter(
        output_directory=tmp_path / "processed",
        allowed_symbols={"BTC-USDT"},
    )

    converter.convert_file(input_file)

    with pytest.raises(FileExistsError):
        converter.convert_file(input_file)


def test_converter_can_overwrite(
    tmp_path: Path,
) -> None:
    input_file = tmp_path / ("tickers-2026-06-27.jsonl")

    write_jsonl(
        input_file,
        [create_record()],
    )

    converter = TickerParquetConverter(
        output_directory=tmp_path / "processed",
        allowed_symbols={"BTC-USDT"},
    )

    converter.convert_file(input_file)
    written_files = converter.convert_file(
        input_file,
        overwrite=True,
    )

    assert len(written_files) == 1
