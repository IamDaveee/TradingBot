import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import structlog

from trading_bot.exchanges.okx.models import OkxTicker
from trading_bot.exchanges.okx.public_websocket import (
    OkxPublicWebSocketClient,
)
from trading_bot.storage.ticker_storage_worker import (
    TickerStorageWorker,
)


class MarketDataUnavailableError(RuntimeError):
    """Raised when requested market data is unavailable."""


class MarketDataService:
    def __init__(
        self,
        websocket_client: OkxPublicWebSocketClient,
        symbols: list[str],
        stale_after_seconds: float,
        storage_worker: TickerStorageWorker | None = None,
    ) -> None:
        if not symbols:
            raise ValueError("At least one market-data symbol is required")

        if stale_after_seconds <= 0:
            raise ValueError("Stale-data threshold must be greater than zero")

        self._websocket_client = websocket_client
        self._symbols = tuple(symbols)
        self._stale_after_seconds = stale_after_seconds
        self._storage_worker = storage_worker

        self._latest_tickers: dict[str, OkxTicker] = {}
        self._lock = asyncio.Lock()
        self._stop_event = asyncio.Event()

        self._logger = structlog.get_logger("market_data_service")

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    async def run(self) -> None:
        self._logger.info(
            "market_data_service_starting",
            symbols=self._symbols,
        )

        self._stop_event.clear()

        ticker_stream = self._websocket_client.stream_tickers(list(self._symbols))

        try:
            await self._consume_tickers(ticker_stream)
        except asyncio.CancelledError:
            self._logger.info("market_data_service_cancelled")
            raise
        finally:
            await ticker_stream.aclose()

            self._logger.info("market_data_service_stopped")

    async def _consume_tickers(
        self,
        ticker_stream: AsyncGenerator[OkxTicker],
    ) -> None:
        async for ticker in ticker_stream:
            if self._stop_event.is_set():
                break

            await self.update_ticker(ticker)

    async def stop(self) -> None:
        self._logger.info("market_data_service_stop_requested")
        self._stop_event.set()

    async def update_ticker(
        self,
        ticker: OkxTicker,
    ) -> None:
        if ticker.instrument_id not in self._symbols:
            self._logger.warning(
                "unexpected_ticker_symbol",
                symbol=ticker.instrument_id,
            )
            return

        async with self._lock:
            self._latest_tickers[ticker.instrument_id] = ticker
        if self._storage_worker is not None:
            await self._storage_worker.enqueue(ticker)

    async def get_latest_ticker(
        self,
        symbol: str,
    ) -> OkxTicker:
        async with self._lock:
            ticker = self._latest_tickers.get(symbol)

        if ticker is None:
            raise MarketDataUnavailableError(f"No ticker data is available for {symbol}")

        return ticker

    async def has_data(
        self,
        symbol: str,
    ) -> bool:
        async with self._lock:
            return symbol in self._latest_tickers

    async def is_ready(self) -> bool:
        async with self._lock:
            return all(symbol in self._latest_tickers for symbol in self._symbols)

    async def get_missing_symbols(self) -> list[str]:
        async with self._lock:
            return [symbol for symbol in self._symbols if symbol not in self._latest_tickers]

    async def is_stale(
        self,
        symbol: str,
        current_time: datetime | None = None,
    ) -> bool:
        ticker = await self.get_latest_ticker(symbol)

        if current_time is None:
            current_time = datetime.now(UTC)

        age_seconds = (current_time - ticker.received_timestamp).total_seconds()

        return age_seconds > self._stale_after_seconds

    async def get_stale_symbols(
        self,
        current_time: datetime | None = None,
    ) -> list[str]:
        if current_time is None:
            current_time = datetime.now(UTC)

        async with self._lock:
            ticker_snapshot = dict(self._latest_tickers)

        stale_symbols: list[str] = []

        for symbol in self._symbols:
            ticker = ticker_snapshot.get(symbol)

            if ticker is None:
                stale_symbols.append(symbol)
                continue

            age_seconds = (current_time - ticker.received_timestamp).total_seconds()

            if age_seconds > self._stale_after_seconds:
                stale_symbols.append(symbol)

        return stale_symbols

    async def get_snapshot(
        self,
    ) -> dict[str, OkxTicker]:
        async with self._lock:
            return dict(self._latest_tickers)
