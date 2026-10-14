import asyncio
import json
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

import structlog
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, WebSocketException

from trading_bot.exchanges.okx.models import OkxTicker


class OkxWebSocketError(RuntimeError):
    """Raised when OKX sends an unsuccessful WebSocket event."""


class OkxPublicWebSocketClient:
    def __init__(
        self,
        websocket_url: str,
        idle_timeout_seconds: float = 20.0,
        pong_timeout_seconds: float = 10.0,
        initial_reconnect_delay_seconds: float = 1.0,
        maximum_reconnect_delay_seconds: float = 30.0,
    ) -> None:
        self._websocket_url = websocket_url
        self._idle_timeout_seconds = idle_timeout_seconds
        self._pong_timeout_seconds = pong_timeout_seconds
        self._initial_reconnect_delay_seconds = initial_reconnect_delay_seconds
        self._maximum_reconnect_delay_seconds = maximum_reconnect_delay_seconds
        self._logger = structlog.get_logger("okx_public_websocket")

    async def stream_tickers(
        self,
        symbols: list[str],
    ) -> AsyncGenerator[OkxTicker]:
        if not symbols:
            raise ValueError("At least one ticker symbol is required")

        reconnect_delay = self._initial_reconnect_delay_seconds

        while True:
            connection_stream = self._stream_tickers_once(symbols)

            try:
                async for ticker in connection_stream:
                    reconnect_delay = self._initial_reconnect_delay_seconds
                    yield ticker

            except asyncio.CancelledError:
                self._logger.info("websocket_stream_cancelled")
                raise

            except (
                ConnectionClosed,
                TimeoutError,
                OSError,
                WebSocketException,
                OkxWebSocketError,
            ) as error:
                self._logger.warning(
                    "websocket_connection_lost",
                    error_type=type(error).__name__,
                    error_message=str(error),
                    reconnect_delay_seconds=reconnect_delay,
                )

                await asyncio.sleep(reconnect_delay)

                reconnect_delay = min(
                    reconnect_delay * 2,
                    self._maximum_reconnect_delay_seconds,
                )

            finally:
                await connection_stream.aclose()

    async def _stream_tickers_once(
        self,
        symbols: list[str],
    ) -> AsyncGenerator[OkxTicker]:
        subscription = {
            "op": "subscribe",
            "args": [
                {
                    "channel": "tickers",
                    "instId": symbol,
                }
                for symbol in symbols
            ],
        }

        self._logger.info(
            "websocket_connecting",
            url=self._websocket_url,
        )

        async with connect(
            self._websocket_url,
            ping_interval=None,
            close_timeout=5,
        ) as websocket:
            self._logger.info("websocket_connected")

            await websocket.send(json.dumps(subscription))

            self._logger.info(
                "websocket_subscription_sent",
                symbols=symbols,
                channel="tickers",
            )

            while True:
                try:
                    raw_message = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=self._idle_timeout_seconds,
                    )

                except TimeoutError:
                    self._logger.warning(
                        "websocket_idle",
                        idle_timeout_seconds=(self._idle_timeout_seconds),
                    )

                    await websocket.send("ping")

                    raw_message = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=self._pong_timeout_seconds,
                    )

                if isinstance(raw_message, bytes):
                    raw_message = raw_message.decode("utf-8")

                if raw_message == "pong":
                    self._logger.debug("websocket_pong_received")
                    continue

                payload: Any = json.loads(raw_message)

                if not isinstance(payload, dict):
                    self._logger.warning(
                        "unexpected_websocket_payload",
                        payload_type=type(payload).__name__,
                    )
                    continue

                event = payload.get("event")

                if event == "subscribe":
                    self._logger.info(
                        "websocket_subscription_confirmed",
                        argument=payload.get("arg"),
                        connection_id=payload.get("connId"),
                    )
                    continue

                if event == "error":
                    code = payload.get("code")
                    message = payload.get("msg")

                    raise OkxWebSocketError(f"OKX WebSocket error {code}: {message}")

                data = payload.get("data")

                if not isinstance(data, list):
                    continue

                for item in data:
                    if not isinstance(item, dict):
                        continue

                    received_timestamp = datetime.now(UTC)

                    yield OkxTicker.from_okx_payload(
                        item,
                        received_timestamp=(received_timestamp),
                    )
