import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from trading_bot.market_data.historical_candle import (
    HistoricalCandle,
)


class OkxHistoricalCandleError(RuntimeError):
    """Raised when OKX historical candle retrieval fails."""


@dataclass(frozen=True, slots=True)
class HistoricalCandlePage:
    candles: list[HistoricalCandle]
    oldest_timestamp_ms: int | None


class OkxHistoricalCandleClient:
    _PATH = "/api/v5/market/history-candles"
    _MAX_PAGE_SIZE = 100

    _INTERVAL_DURATIONS = {
        "1m": timedelta(minutes=1),
        "3m": timedelta(minutes=3),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1H": timedelta(hours=1),
        "2H": timedelta(hours=2),
        "4H": timedelta(hours=4),
        "6H": timedelta(hours=6),
        "12H": timedelta(hours=12),
        "1D": timedelta(days=1),
    }

    def __init__(
        self,
        base_url: str = "https://www.okx.com",
        timeout_seconds: float = 15.0,
        maximum_attempts: int = 5,
        retry_base_delay_seconds: float = 0.5,
        request_delay_seconds: float = 0.12,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        if maximum_attempts <= 0:
            raise ValueError("maximum_attempts must be greater than zero")

        if retry_base_delay_seconds < 0:
            raise ValueError("retry_base_delay_seconds must not be negative")

        if request_delay_seconds < 0:
            raise ValueError("request_delay_seconds must not be negative")

        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._maximum_attempts = maximum_attempts
        self._retry_base_delay_seconds = retry_base_delay_seconds
        self._request_delay_seconds = request_delay_seconds

        self._external_client = client
        self._owned_client: httpx.AsyncClient | None = None

    async def __aenter__(
        self,
    ) -> "OkxHistoricalCandleClient":
        if self._external_client is None:
            self._owned_client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout_seconds,
                headers={
                    "Accept": "application/json",
                    "User-Agent": ("crypto-trade-bot/1.0"),
                },
            )

        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        if self._owned_client is not None:
            await self._owned_client.aclose()
            self._owned_client = None

    async def fetch_page(
        self,
        instrument_id: str,
        interval: str,
        *,
        after_timestamp_ms: int | None = None,
        before_timestamp_ms: int | None = None,
        limit: int = 100,
        include_incomplete: bool = False,
    ) -> HistoricalCandlePage:
        if not instrument_id:
            raise ValueError("instrument_id must not be empty")

        if interval not in self._INTERVAL_DURATIONS:
            raise ValueError(f"Unsupported interval: {interval}")

        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        if limit > self._MAX_PAGE_SIZE:
            raise ValueError(f"limit must not exceed {self._MAX_PAGE_SIZE}")

        params: dict[str, str] = {
            "instId": instrument_id,
            "bar": interval,
            "limit": str(limit),
        }

        if after_timestamp_ms is not None:
            params["after"] = str(after_timestamp_ms)

        if before_timestamp_ms is not None:
            params["before"] = str(before_timestamp_ms)

        payload = await self._request(params)

        raw_rows = payload.get("data")

        if not isinstance(raw_rows, list):
            raise OkxHistoricalCandleError("OKX response data must be a list")

        candles: list[HistoricalCandle] = []

        oldest_timestamp_ms: int | None = None

        for raw_row in raw_rows:
            candle = self._parse_row(
                instrument_id=instrument_id,
                interval=interval,
                raw_row=raw_row,
            )

            timestamp_ms = int(candle.interval_start.timestamp() * 1000)

            if oldest_timestamp_ms is None or timestamp_ms < oldest_timestamp_ms:
                oldest_timestamp_ms = timestamp_ms

            if include_incomplete or candle.is_complete:
                candles.append(candle)

        candles.sort(key=lambda candle: candle.interval_start)

        return HistoricalCandlePage(
            candles=candles,
            oldest_timestamp_ms=(oldest_timestamp_ms),
        )

    async def fetch_range(
        self,
        instrument_id: str,
        interval: str,
        start: datetime,
        end: datetime,
        *,
        include_incomplete: bool = False,
    ) -> list[HistoricalCandle]:
        self._validate_range(
            start=start,
            end=end,
        )

        start_utc = start.astimezone(UTC)
        end_utc = end.astimezone(UTC)

        start_ms = self._datetime_to_ms(start_utc)

        end_ms = self._datetime_to_ms(end_utc)

        cursor_ms = end_ms

        candles_by_timestamp: dict[
            datetime,
            HistoricalCandle,
        ] = {}

        previous_oldest_ms: int | None = None

        while True:
            page = await self.fetch_page(
                instrument_id=instrument_id,
                interval=interval,
                after_timestamp_ms=cursor_ms,
                limit=self._MAX_PAGE_SIZE,
                include_incomplete=(include_incomplete),
            )

            if page.oldest_timestamp_ms is None:
                break

            for candle in page.candles:
                if start_utc <= candle.interval_start < end_utc:
                    candles_by_timestamp[candle.interval_start] = candle

            if page.oldest_timestamp_ms <= start_ms:
                break

            if previous_oldest_ms == page.oldest_timestamp_ms:
                raise OkxHistoricalCandleError("Pagination cursor did not advance")

            previous_oldest_ms = page.oldest_timestamp_ms

            cursor_ms = page.oldest_timestamp_ms

            if self._request_delay_seconds > 0:
                await asyncio.sleep(self._request_delay_seconds)

        return sorted(
            candles_by_timestamp.values(),
            key=lambda candle: candle.interval_start,
        )

    async def _request(
        self,
        params: dict[str, str],
    ) -> dict[str, Any]:
        client = self._get_client()

        last_error: Exception | None = None

        for attempt in range(
            1,
            self._maximum_attempts + 1,
        ):
            try:
                response = await client.get(
                    self._PATH,
                    params=params,
                )

                response.raise_for_status()

                payload = response.json()

                if not isinstance(payload, dict):
                    raise OkxHistoricalCandleError("OKX response must be an object")

                code = payload.get("code")

                if code != "0":
                    message = payload.get(
                        "msg",
                        "Unknown OKX API error",
                    )

                    raise OkxHistoricalCandleError(f"OKX API error code={code}: {message}")

                return payload

            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.HTTPStatusError,
                ValueError,
                OkxHistoricalCandleError,
            ) as error:
                last_error = error

                if attempt >= self._maximum_attempts:
                    break

                delay = self._retry_base_delay_seconds * (2 ** (attempt - 1))

                if delay > 0:
                    await asyncio.sleep(delay)

        raise OkxHistoricalCandleError(
            f"Historical candle request failed after {self._maximum_attempts} attempts"
        ) from last_error

    def _parse_row(
        self,
        instrument_id: str,
        interval: str,
        raw_row: object,
    ) -> HistoricalCandle:
        if not isinstance(raw_row, Sequence):
            raise OkxHistoricalCandleError("Candle row must be a sequence")

        if isinstance(
            raw_row,
            (str, bytes, bytearray),
        ):
            raise OkxHistoricalCandleError("Candle row must not be a string")

        if len(raw_row) < 9:
            raise OkxHistoricalCandleError("Candle row must contain at least nine values")

        try:
            timestamp_ms = int(raw_row[0])

            interval_start = datetime.fromtimestamp(
                timestamp_ms / 1000,
                tz=UTC,
            )

            interval_end = interval_start + self._INTERVAL_DURATIONS[interval]

            return HistoricalCandle(
                instrument_id=instrument_id,
                interval=interval,
                interval_start=interval_start,
                interval_end=interval_end,
                open_price=Decimal(str(raw_row[1])),
                high_price=Decimal(str(raw_row[2])),
                low_price=Decimal(str(raw_row[3])),
                close_price=Decimal(str(raw_row[4])),
                base_volume=Decimal(str(raw_row[5])),
                currency_volume=Decimal(str(raw_row[6])),
                quote_volume=Decimal(str(raw_row[7])),
                is_complete=(str(raw_row[8]) == "1"),
            )
        except (
            IndexError,
            TypeError,
            ValueError,
            InvalidOperation,
        ) as error:
            raise OkxHistoricalCandleError(f"Invalid historical candle row: {raw_row!r}") from error

    def _get_client(
        self,
    ) -> httpx.AsyncClient:
        client = self._external_client or self._owned_client

        if client is None:
            raise RuntimeError("Use OkxHistoricalCandleClient inside an async context manager")

        return client

    @staticmethod
    def _validate_range(
        start: datetime,
        end: datetime,
    ) -> None:
        if start.tzinfo is None:
            raise ValueError("start must contain a timezone")

        if end.tzinfo is None:
            raise ValueError("end must contain a timezone")

        if end <= start:
            raise ValueError("end must be after start")

    @staticmethod
    def _datetime_to_ms(
        value: datetime,
    ) -> int:
        return int(value.timestamp() * 1000)
