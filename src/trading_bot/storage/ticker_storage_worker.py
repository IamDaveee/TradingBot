import asyncio

import structlog

from trading_bot.exchanges.okx.models import OkxTicker
from trading_bot.storage.jsonl_writer import JsonlTickerWriter


class TickerStorageWorker:
    def __init__(
        self,
        writer: JsonlTickerWriter,
        queue_max_size: int,
    ) -> None:
        if queue_max_size <= 0:
            raise ValueError("queue_max_size must be greater than zero")

        self._writer = writer

        self._queue: asyncio.Queue[OkxTicker | None] = asyncio.Queue(
            maxsize=queue_max_size,
        )

        self._ready_event = asyncio.Event()

        self._started = False
        self._stopping = False
        self._records_written = 0

        self._logger = structlog.get_logger("ticker_storage_worker")

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def queue_capacity(self) -> int:
        return self._queue.maxsize

    @property
    def records_written(self) -> int:
        return self._records_written

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def is_stopping(self) -> bool:
        return self._stopping

    async def wait_until_started(self) -> None:
        await self._ready_event.wait()

    async def enqueue(
        self,
        ticker: OkxTicker,
    ) -> None:
        if not self._started:
            raise RuntimeError("Ticker storage worker has not started")

        if self._stopping:
            raise RuntimeError("Ticker storage worker is stopping")

        await self._queue.put(ticker)

    async def run(self) -> None:
        if self._started:
            raise RuntimeError("Ticker storage worker is already running")

        self._started = True
        self._stopping = False
        self._ready_event.set()

        self._logger.info(
            "ticker_storage_worker_started",
            queue_capacity=self.queue_capacity,
        )

        try:
            while True:
                item = await self._queue.get()

                try:
                    if item is None:
                        break

                    await self._writer.write(item)
                    self._records_written += 1

                finally:
                    self._queue.task_done()

        except asyncio.CancelledError:
            self._logger.warning(
                "ticker_storage_worker_cancelled",
                queue_size=self.queue_size,
            )
            raise

        finally:
            await self._writer.close()

            self._started = False
            self._stopping = False
            self._ready_event.clear()

            self._logger.info(
                "ticker_storage_worker_stopped",
                records_written=self._records_written,
                remaining_queue_size=self.queue_size,
            )

    async def stop(self) -> None:
        if not self._started:
            return

        if self._stopping:
            return

        self._stopping = True

        self._logger.info(
            "ticker_storage_worker_stop_requested",
            queue_size=self.queue_size,
        )

        await self._queue.put(None)

    async def wait_until_empty(self) -> None:
        await self._queue.join()
