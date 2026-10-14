import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from trading_bot.exchanges.okx.models import OkxTicker
from trading_bot.storage.jsonl_writer import (
    JsonlTickerWriter,
)


def create_ticker(
    received_timestamp: datetime,
) -> OkxTicker:
    return OkxTicker(
        instrument_id="BTC-USDT",
        last_price=Decimal("60000.1"),
        last_size=Decimal("0.01"),
        best_ask_price=Decimal("60000.2"),
        best_ask_size=Decimal("1.2"),
        best_bid_price=Decimal("60000.1"),
        best_bid_size=Decimal("1.1"),
        exchange_timestamp=received_timestamp,
        received_timestamp=received_timestamp,
    )


@pytest.mark.asyncio
async def test_writer_creates_jsonl_file(
    tmp_path: Path,
) -> None:
    writer = JsonlTickerWriter(
        directory=tmp_path,
        flush_every_records=1,
    )

    timestamp = datetime(
        2026,
        6,
        27,
        12,
        0,
        tzinfo=UTC,
    )

    await writer.write(create_ticker(timestamp))
    await writer.close()

    file_path = tmp_path / "tickers-2026-06-27.jsonl"

    assert file_path.exists()

    lines = file_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1

    record = json.loads(lines[0])

    assert record["instrument_id"] == "BTC-USDT"
    assert record["last_price"] == "60000.1"
    assert record["spread"] == "0.1"


@pytest.mark.asyncio
async def test_writer_appends_multiple_records(
    tmp_path: Path,
) -> None:
    writer = JsonlTickerWriter(
        directory=tmp_path,
        flush_every_records=2,
    )

    timestamp = datetime(
        2026,
        6,
        27,
        12,
        0,
        tzinfo=UTC,
    )

    ticker = create_ticker(timestamp)

    await writer.write(ticker)
    await writer.write(ticker)
    await writer.close()

    file_path = tmp_path / "tickers-2026-06-27.jsonl"

    lines = file_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2


def test_writer_rejects_invalid_flush_interval(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        JsonlTickerWriter(
            directory=tmp_path,
            flush_every_records=0,
        )
