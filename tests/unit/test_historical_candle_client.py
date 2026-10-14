from datetime import UTC, datetime

import httpx
import pytest

from trading_bot.exchanges.okx.historical_candle_client import (
    OkxHistoricalCandleClient,
)


@pytest.mark.asyncio
async def test_fetch_page_parses_okx_rows() -> None:
    timestamp_ms = 1780315200000

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        assert request.url.params["instId"] == "BTC-USDT"

        assert request.url.params["bar"] == "1m"

        return httpx.Response(
            200,
            json={
                "code": "0",
                "msg": "",
                "data": [
                    [
                        str(timestamp_ms),
                        "100",
                        "110",
                        "90",
                        "105",
                        "12.5",
                        "12.5",
                        "1300",
                        "1",
                    ]
                ],
            },
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        base_url="https://www.okx.com",
        transport=transport,
    ) as http_client:
        async with OkxHistoricalCandleClient(client=http_client) as client:
            page = await client.fetch_page(
                instrument_id="BTC-USDT",
                interval="1m",
            )

    assert len(page.candles) == 1

    candle = page.candles[0]

    assert candle.open_price == 100
    assert candle.close_price == 105
    assert candle.is_complete

    assert candle.interval_start == datetime.fromtimestamp(
        timestamp_ms / 1000,
        tz=UTC,
    )


@pytest.mark.asyncio
async def test_excludes_incomplete_candles() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "code": "0",
                "msg": "",
                "data": [
                    [
                        "1780315200000",
                        "100",
                        "110",
                        "90",
                        "105",
                        "12.5",
                        "12.5",
                        "1300",
                        "0",
                    ]
                ],
            },
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        base_url="https://www.okx.com",
        transport=transport,
    ) as http_client:
        async with OkxHistoricalCandleClient(client=http_client) as client:
            page = await client.fetch_page(
                instrument_id="BTC-USDT",
                interval="1m",
            )

    assert page.candles == []

    assert page.oldest_timestamp_ms == 1780315200000


@pytest.mark.asyncio
async def test_rejects_api_error() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "code": "51000",
                "msg": "Parameter error",
                "data": [],
            },
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        base_url="https://www.okx.com",
        transport=transport,
    ) as http_client:
        client = OkxHistoricalCandleClient(
            client=http_client,
            maximum_attempts=1,
        )

        with pytest.raises(RuntimeError):
            await client.fetch_page(
                instrument_id="BTC-USDT",
                interval="1m",
            )


@pytest.mark.asyncio
async def test_rejects_page_size_above_limit() -> None:
    async with httpx.AsyncClient(base_url="https://www.okx.com") as http_client:
        client = OkxHistoricalCandleClient(client=http_client)

        with pytest.raises(ValueError):
            await client.fetch_page(
                instrument_id="BTC-USDT",
                interval="1m",
                limit=101,
            )
