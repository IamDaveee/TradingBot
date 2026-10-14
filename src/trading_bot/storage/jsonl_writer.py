import asyncio
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TextIO

import structlog

from trading_bot.exchanges.okx.models import OkxTicker


class JsonlTickerWriter:
    def __init__(
        self,
        directory: Path,
        flush_every_records: int = 10,
    ) -> None:
        if flush_every_records <= 0:
            raise ValueError("flush_every_records must be greater than zero")

        self._directory = directory
        self._flush_every_records = flush_every_records

        self._file: TextIO | None = None
        self._file_date: date | None = None
        self._records_since_flush = 0

        self._lock = asyncio.Lock()
        self._logger = structlog.get_logger("jsonl_ticker_writer")

    async def write(
        self,
        ticker: OkxTicker,
    ) -> None:
        async with self._lock:
            self._ensure_file_open(ticker.received_timestamp.date())

            if self._file is None:
                raise RuntimeError("Ticker output file is not open")

            record = self._serialize_ticker(ticker)

            self._file.write(
                json.dumps(
                    record,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            )
            self._file.write("\n")

            self._records_since_flush += 1

            if self._records_since_flush >= self._flush_every_records:
                self._file.flush()
                self._records_since_flush = 0

    async def close(self) -> None:
        async with self._lock:
            if self._file is None:
                return

            self._file.flush()
            self._file.close()

            self._logger.info(
                "ticker_storage_file_closed",
                date=str(self._file_date),
            )

            self._file = None
            self._file_date = None
            self._records_since_flush = 0

    def _ensure_file_open(
        self,
        record_date: date,
    ) -> None:
        if self._file is not None and self._file_date == record_date:
            return

        if self._file is not None:
            self._file.flush()
            self._file.close()

        self._directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_path = self._directory / f"tickers-{record_date.isoformat()}.jsonl"

        self._file = file_path.open(
            mode="a",
            encoding="utf-8",
            newline="\n",
        )
        self._file_date = record_date
        self._records_since_flush = 0

        self._logger.info(
            "ticker_storage_file_opened",
            path=str(file_path),
        )

    @staticmethod
    def _serialize_ticker(
        ticker: OkxTicker,
    ) -> dict[str, str]:
        return {
            "instrument_id": ticker.instrument_id,
            "last_price": str(ticker.last_price),
            "last_size": str(ticker.last_size),
            "best_ask_price": str(ticker.best_ask_price),
            "best_ask_size": str(ticker.best_ask_size),
            "best_bid_price": str(ticker.best_bid_price),
            "best_bid_size": str(ticker.best_bid_size),
            "spread": str(ticker.spread),
            "spread_percentage": str(ticker.spread_percentage),
            "exchange_timestamp": (ticker.exchange_timestamp.astimezone(UTC).isoformat()),
            "received_timestamp": (ticker.received_timestamp.astimezone(UTC).isoformat()),
            "written_timestamp": (datetime.now(UTC).isoformat()),
        }
