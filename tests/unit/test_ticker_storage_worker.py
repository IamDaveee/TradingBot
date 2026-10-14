import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from trading_bot.exchanges.okx.models import OkxTicker
from trading_bot.storage.jsonl_writer import (
    JsonlTickerWriter,
)
from trading_bot.storage.ticker_storage_worker import (
    TickerStorageWorker,
)


def create_ticker() -> OkxTicker:
    timestamp = datetime(
        2026,
        6,
        27,
        12,
        0,
        tzinfo=UTC,
    )

    return OkxTicker(
        instrument_id="BTC-USDT",
        last_price=Decimal("60000"),
        last_size=Decimal("0.01"),
        best_ask_price=Decimal("60000.1"),
        best_ask_size=Decimal("1"),
        best_bid_price=Decimal("60000"),
        best_bid_size=Decimal("1"),
        exchange_timestamp=timestamp,
        received_timestamp=timestamp,
    )


@pytest.mark.asyncio
async def test_worker_writes_queued_ticker(
    tmp_path: Path,
) -> None:
    writer = JsonlTickerWriter(
        directory=tmp_path,
        flush_every_records=1,
    )

    worker = TickerStorageWorker(
        writer=writer,
        queue_max_size=10,
    )

    task = asyncio.create_task(worker.run())
    await worker.wait_until_started()

    await worker.enqueue(create_ticker())
    await worker.wait_until_empty()
    await worker.stop()
    await task

    assert worker.records_written == 1

    file_path = tmp_path / "tickers-2026-06-27.jsonl"

    assert file_path.exists()


@pytest.mark.asyncio
async def test_worker_rejects_enqueue_before_start(
    tmp_path: Path,
) -> None:
    writer = JsonlTickerWriter(
        directory=tmp_path,
    )

    worker = TickerStorageWorker(
        writer=writer,
        queue_max_size=10,
    )

    with pytest.raises(RuntimeError):
        await worker.enqueue(create_ticker())


def test_worker_rejects_invalid_queue_size(
    tmp_path: Path,
) -> None:
    writer = JsonlTickerWriter(
        directory=tmp_path,
    )

    with pytest.raises(ValueError):
        TickerStorageWorker(
            writer=writer,
            queue_max_size=0,
        )
